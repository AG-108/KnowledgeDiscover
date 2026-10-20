# User-facing DLGA wrapper with a configurable operator library.
import heapq
import logging
import os
import pickle
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from kd.model.dlga import DLGA


class KD_DLGA(DLGA):
    """Expose DLGA with a configurable operator library and benchmark-friendly outputs."""

    def __init__(
        self, operators: list[str], epi: float, input_dim: int, verbose: bool = True,
        library_kind=None, coordinate_terms=False, normalize_coordinates=False,
        fitness_kind="legacy", **kwargs
    ):
        self.user_operators = operators
        self.library_size = len(operators)
        self.verbose = verbose
        super().__init__(epi=epi, input_dim=input_dim, **kwargs)
        self.equations_ = None
        self.best_equation_ = None
        self.library_kind = library_kind
        self.coordinate_terms = coordinate_terms
        self.normalize_coordinates = normalize_coordinates
        self.fitness_kind = fitness_kind

    def train_NN(self, X, y):
        if not self.normalize_coordinates:
            return super().train_NN(X, y)
        import copy
        from kd.benchmark_deadline import current_deadline, SoftDeadline

        class ScaledNetwork(torch.nn.Module):
            def __init__(self, network, values, targets):
                super().__init__()
                self.network = network
                self.register_buffer("center", torch.as_tensor((values.max(0) + values.min(0)) / 2))
                self.register_buffer("scale", torch.as_tensor(np.maximum(np.ptp(values, axis=0) / 2, 1e-8)))
                self.register_buffer("offset", torch.as_tensor(float(np.mean(targets))))
                self.register_buffer("amplitude", torch.as_tensor(max(float(np.std(targets)), 1e-8)))

            def forward(self, values):
                return self.offset + self.amplitude * self.network((values - self.center) / self.scale)

        self.Net = ScaledNetwork(self.Net, X, y).to(device=self.device, dtype=self.dtype)
        permutation = np.random.permutation(len(X))
        ntrain = max(2, int(len(X) * 0.9))
        tx = torch.as_tensor(X[permutation], device=self.device, dtype=self.dtype)
        ty = torch.as_tensor(y[permutation], device=self.device, dtype=self.dtype).reshape(-1, 1)
        deadline = current_deadline()
        stop_at = time.perf_counter() + deadline.remaining() * 0.7 if deadline else float("inf")
        best, best_state = float("inf"), copy.deepcopy(self.Net.state_dict())
        optimizer = torch.optim.Adam(self.Net.parameters(), lr=self.learning_rate)

        def loss():
            if time.perf_counter() >= stop_at:
                raise SoftDeadline()
            return torch.mean((self.Net(tx[:ntrain]) - ty[:ntrain])**2) / self.Net.amplitude**2

        def checkpoint():
            nonlocal best, best_state
            with torch.no_grad():
                error = float(torch.mean((self.Net(tx[ntrain:]) - ty[ntrain:])**2))
            self.val_loss_history.append(error)
            if error < best:
                best, best_state = error, copy.deepcopy(self.Net.state_dict())

        try:
            for step in range(self.max_iter):
                optimizer.zero_grad()
                value = loss()
                if not torch.isfinite(value):
                    raise RuntimeError("Nonfinite DLGA observation loss")
                value.backward()
                optimizer.step()
                if step % 100 == 0:
                    self.train_loss_history.append(float(value.detach()))
                    checkpoint()
            checkpoint()
            if self.least_squares_nfev:
                self.Net.load_state_dict(best_state)
                self._refine_observations(tx[:ntrain], ty[:ntrain])
                checkpoint()
            elif self.lbfgs_max_iter:
                self.Net.load_state_dict(best_state)
                optimizer = torch.optim.LBFGS(self.Net.parameters(), max_iter=self.lbfgs_max_iter,
                    tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
                calls = [0]

                def closure():
                    optimizer.zero_grad()
                    value = loss()
                    value.backward()
                    calls[0] += 1
                    if calls[0] % 25 == 0:
                        checkpoint()
                    return value

                optimizer.step(closure)
                checkpoint()
        except SoftDeadline:
            checkpoint()
        self.Net.load_state_dict(best_state)
        self.best_epoch = "calibrated"
        os.makedirs("model_save", exist_ok=True)
        torch.save(best_state, "model_save/Net_calibrated.pkl")
        self.observation_validation_nmse_ = best / float(self.Net.amplitude**2)
        return self.Net, self.best_epoch

    def generate_meta_data(self, X):
        self.Net.to(self.device)
        X_tensor = torch.as_tensor(np.asarray(X, dtype=self.numpy_dtype), device=self.device)
        X_tensor.requires_grad_(True)
        self.Net.load_state_dict(
            torch.load(
                f"model_save/Net_{self.best_epoch}.pkl", map_location=self.device, weights_only=True
            )
        )
        self.Net.eval()
        u = self.Net(X_tensor)
        u_grad = torch.autograd.grad(outputs=u.sum(), inputs=X_tensor, create_graph=True)[0]
        ux = u_grad[:, 0].reshape(-1, 1) if u_grad.shape[1] > 0 else torch.zeros_like(u)
        ut = u_grad[:, 1].reshape(-1, 1) if u_grad.shape[1] > 1 else torch.zeros_like(u)
        try:
            uxx_grad = torch.autograd.grad(outputs=ux.sum(), inputs=X_tensor, create_graph=True)[0]
            uxx = uxx_grad[:, 0].reshape(-1, 1)
            try:
                uxxx_grad = torch.autograd.grad(
                    outputs=uxx.sum(), inputs=X_tensor, create_graph=True
                )[0]
                uxxx = uxxx_grad[:, 0].reshape(-1, 1)
            except Exception as e:
                logging.warning(
                    "Failed to compute u_xxx via autograd, falling back to zeros: %s", e
                )
                uxxx = torch.zeros_like(u)
        except Exception as e:
            logging.warning(
                "Failed to compute u_xx/u_xxx via autograd, falling back to zeros: %s", e
            )
            uxx = torch.zeros_like(u)
            uxxx = torch.zeros_like(u)
        try:
            utt_grad = torch.autograd.grad(outputs=ut.sum(), inputs=X_tensor, create_graph=True)[0]
            utt = utt_grad[:, 1].reshape(-1, 1)
        except Exception as e:
            logging.warning("Failed to compute u_tt via autograd, falling back to zeros: %s", e)
            utt = torch.zeros_like(u)
        available_ops = {"u": u, "u_x": ux, "u_t": ut, "u_xx": uxx, "u_xxx": uxxx, "u_tt": utt}
        if self.library_kind:
            from kd.model.pde_library import evaluate_terms
            available_ops.update(zip(self.user_operators, evaluate_terms(
                self.user_operators, {**available_ops, "x": X_tensor[:, :1]}
            )))
        # Restrict Theta to the configured operators used by the equation search.
        theta_columns = []
        for op_name in self.user_operators:
            if op_name in available_ops:
                theta_columns.append(available_ops[op_name])
        self.Theta = (
            torch.concatenate(theta_columns, axis=1)
            if theta_columns
            else torch.empty(X.shape[0], 0).to(self.device)
        )

        # Keep every computed derivative in metadata for diagnostics and visualization.
        self.metadata = {key: val.cpu().detach().numpy() for key, val in available_ops.items()}

        # Preserve the derivative aliases expected by the parent implementation.
        self.u_t = ut.cpu().detach().numpy()
        self.u_tt = utt.cpu().detach().numpy()

        return self.Theta

    def random_module(self):
        if self.library_kind:
            return [random.randrange(self.library_size)]
        genes_module = []
        for _ in range(self.max_length):
            if self.library_size == 0:
                break
            a = random.randint(0, self.library_size - 1)
            genes_module.append(a)
            if random.uniform(0, 1) > self.partial_prob:
                break
        return genes_module

    def mutation(self):
        Chrom, size_pop = self.Chrom, self.pop_size
        for i in range(size_pop):
            if random.uniform(0, 1) < self.add_rate:
                add_Chrom = self.random_module()
                if add_Chrom and add_Chrom not in Chrom[i]:
                    Chrom[i].append(add_Chrom)
            if random.uniform(0, 1) < self.delete_rate and len(Chrom[i]) > 1:
                Chrom[i].pop(random.randint(0, len(Chrom[i]) - 1))
            if random.uniform(0, 1) < self.mutate_rate and self.library_size > 0:
                if Chrom[i]:
                    try:
                        n1 = random.randint(0, len(Chrom[i]) - 1)
                        if Chrom[i][n1]:
                            n2 = random.randint(0, len(Chrom[i][n1]) - 1)
                            Chrom[i][n1][n2] = random.randint(0, self.library_size - 1)
                    except ValueError:
                        continue
        self.Chrom = Chrom
        return self.Chrom

    def random_genome(self):
        genes = []
        for _ in range(self.max_length):
            gene_random = self.random_module()
            if gene_random:
                genes.append(sorted(gene_random))
            if random.uniform(0, 1) > self.genes_prob:
                break
        return genes

    def convert_chrom_to_eq(self, chrom, left_name, coef):
        name = self.user_operators
        string = []
        if not chrom:
            return f"{left_name}= (empty equation)"
        for i in range(len(chrom)):
            if i >= len(coef):
                break
            item = chrom[i]
            # Preserve the fitted coefficient; display rounding changes the
            # exported equation and prevents reproducible precision checks.
            coef_str = str(float(coef[i, 0]))
            term_str = "*".join([
                f"({name[gene]})" if self.library_kind else name[gene]
                for gene in item if gene < len(name)
            ])
            if not term_str:
                continue
            string.append(f"{coef_str}*{term_str}")
        equation = f"{left_name}=" + "+".join(string).replace("+-", "-")
        return equation

    def evolution(self):
        self.Chrom = []
        self.Fitness = []
        for _ in range(self.pop_size):
            initial_genome = self.random_genome()
            self.Chrom.append(initial_genome)
            if not initial_genome:
                self.Fitness.append(float("inf"))
                continue
            gene_translate, length_penalty_coef = self.translate_DNA(initial_genome)
            _, MSE, _, _ = self.get_fitness(gene_translate, length_penalty_coef)
            self.Fitness.append(MSE)
        self.delete_duplicates()
        os.makedirs("result_save/", exist_ok=True)

        for iter_num in tqdm(range(self.n_generations)):
            from kd.benchmark_deadline import current_deadline
            deadline = current_deadline()
            if deadline and deadline.remaining() < 2 and hasattr(self, "coef"):
                self.stop_reason_ = "time_limit"
                break
            best_chrom_path = "result_save/best_save.pkl"
            best = []
            if self.Chrom:
                best = self.Chrom[0]
                with open(best_chrom_path, "wb") as f:
                    pickle.dump(best, f)
            self.cross_over()
            self.mutation()
            self.delete_duplicates()
            if os.path.exists(best_chrom_path):
                with open(best_chrom_path, "rb") as f:
                    best_loaded = pickle.load(f)
                if best_loaded:
                    if self.Chrom:
                        self.Chrom[0] = best_loaded
                    else:
                        self.Chrom.append(best_loaded)
            self.select()
            if self.verbose and self.Chrom and self.Chrom[0] != best:
                output_str = (
                    f"iter: {iter_num + 1}\n"
                    f"The best Chrom: {self.Chrom[0]}\n"
                    f"The best coef:  \n{self.coef[0]}\n"
                    f"The best fitness: {self.Fitness[0]}\n"
                    f"The best name: {self.name[0]}\n"
                    "----------------------------------------\n"
                )
                with open("result_save/DLGA_output.txt", "a") as f:
                    f.write(output_str)
                print(output_str)
        if not self.Chrom:
            print("Evolution finished, but no solution was found.")
            return [], [], float("inf"), ""
        return self.Chrom[0], self.coef[0], self.Fitness[0], self.name[0]

    def select(self):
        # Re-score the population before retaining the best half.
        Chrom, size_pop = self.Chrom, self.pop_size
        new_Chrom, new_fitness, new_coef, new_name = [], [], [], []
        fitness_list, coef_list, name_list = [], [], []
        for i in range(size_pop):
            if not Chrom[i]:
                fitness_list.append(float("inf"))
                coef_list.append([])
                name_list.append("")
                continue
            gene_translate, length_penalty_coef = self.translate_DNA(Chrom[i])
            coef, MSE, _, name = self.get_fitness(gene_translate, length_penalty_coef)
            fitness_list.append(MSE)
            coef_list.append(coef)
            name_list.append(name)
        num_to_keep = int(size_pop / 2)
        if len(fitness_list) < num_to_keep:
            num_to_keep = len(fitness_list)
        re1 = list(map(fitness_list.index, heapq.nsmallest(num_to_keep, fitness_list)))
        for index in re1:
            new_Chrom.append(Chrom[index])
            new_fitness.append(fitness_list[index])
            new_coef.append(coef_list[index])
            new_name.append(name_list[index])
        for _ in range(size_pop - len(new_Chrom)):
            new = self.random_genome()
            new_Chrom.append(new)
        self.Chrom = new_Chrom
        self.Fitness = new_fitness
        self.coef = new_coef
        self.name = new_name

        # Empty populations cannot contribute fitness or complexity diagnostics.
        if self.Fitness and self.Chrom:
            # Record the best fitness for this generation.
            if hasattr(self, "fitness_history"):
                self.fitness_history.append(min(self.Fitness))

            # Store generation diagnostics consumed by the visualization helpers.
            if hasattr(self, "evolution_history"):
                unique_genes = len(
                    set(tuple(sorted(module)) for chrom in self.Chrom if chrom for module in chrom)
                )
                self.evolution_history.append(
                    {
                        "generation": len(self.fitness_history),
                        "fitness": min(self.Fitness),
                        "complexity": len(self.Chrom[0]) if self.Chrom[0] else 0,
                        "population_size": self.pop_size,
                        "unique_modules": unique_genes,
                    }
                )

        return self.Chrom, self.Fitness, self.coef, self.name

    def fit(self, X, y):
        """Fit the neural surrogate and run the operator-aware genetic search."""
        if self.library_kind:
            from kd.model.pde_library import library_names
            self.user_operators = library_names(self.library_kind, self.coordinate_terms, X[:, 0])
            self.library_size = len(self.user_operators)
        self.Net, self.best_epoch = self.train_NN(X, y)
        self.Theta = self.generate_meta_data(X)
        self._theta_numpy = self.Theta.detach().cpu().numpy()
        self.stop_reason_ = "budget_exhausted"

        try:
            # Run the operator-aware evolutionary search.
            Chrom, coef, fitness, name = self.evolution()

            if not Chrom or (isinstance(coef, np.ndarray) and coef.size == 0):
                fallback_name = name if isinstance(name, str) else "u_t"
                self.best_equation_ = f"{fallback_name}= (empty equation)"
                self.equations_ = [self.best_equation_]
                return self

            print("\nFinal solution debug info:")
            print(f"Chromosome length: {len(Chrom)}")
            print(f"Coefficient shape: {coef.shape}")
            print(f"Chromosome: {Chrom}")
            print(f"Coefficients: {coef}")

            # Convert genes with the configured operator names.
            equation = self.convert_chrom_to_eq(Chrom, name, coef)
            self.best_equation_ = equation
            self.equations_ = [equation]
            print(f"equation form: {equation}")

            # LaTeX rendering is optional and should not invalidate a fitted model.
            try:
                from kd.viz import dlga_eq2latex as equation_renderer

                # Pass the dynamic operator mapping to the renderer.
                self.eq_latex = equation_renderer.dlga_eq2latex(
                    chromosome=Chrom,
                    coefficients=coef,
                    lhs_name_str=name,
                    operator_names=self.user_operators,  # Preserve the model-specific operator order.
                )
            except ImportError:
                print(
                    "\n[INFO] Equation renderer not found or failed to import. Skipping LaTeX generation."
                )

        except Exception as e:
            print(f"Error in fit: {str(e)}")
            # Include a traceback because search failures otherwise lose their context.
            import traceback

            traceback.print_exc()
            raise

    def predict(self, mesh_data):
        X_tensor = torch.from_numpy(mesh_data.astype(self.numpy_dtype)).to(
            self.device
        )  # Convert to tensor and predict
        with torch.no_grad():
            u_pred = self.Net(X_tensor).cpu().numpy()
        return u_pred

    def cross_over(self):
        # Population length and generation budget are independent parameters.
        for index in range(0, len(self.Chrom) - 1, 2):
            left, right = self.Chrom[index:index + 2]
            if left and right:
                i, j = random.randrange(len(left)), random.randrange(len(right))
                left[i], right[j] = right[j].copy(), left[i].copy()
        return self.Chrom

    def translate_DNA(self, gene):
        if not hasattr(self, "_theta_numpy"):
            return super().translate_DNA(gene)
        columns = [np.prod(self._theta_numpy[:, module], axis=1) for module in gene]
        return np.column_stack(columns), sum(map(len, gene))

    def get_fitness(self, gene_translate, length_penalty_coef):
        if self.fitness_kind == "legacy":
            return super().get_fitness(gene_translate, length_penalty_coef)
        target = self.u_t.reshape(-1)
        scales = np.linalg.norm(gene_translate, axis=0)
        safe = np.maximum(scales, 1e-15)
        design = gene_translate / safe
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0] / safe
        mse = float(np.mean((target - gene_translate @ coefficients)**2))
        normalized = mse / max(float(np.mean(target**2)), 1e-30)
        return coefficients[:, None], normalized + self.epi * length_penalty_coef, mse, "u_t"
