# kd/model/kd_eqgpt.py

import random
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from ..base import BaseEstimator
from .eqgpt import (
    GPT,
    NN,
    calculate_reward,
    find_min_no_repeat,
    generate_meta_grid,
    get_mask_invalid,
    get_train_dataset,
    pretrain_gpt,
    random_data,
    sentence_to_str,
    set_device,
    train_surrogate_model,
    word2id,
)

# Exact "Equation name" strings in the vendored PDE_dataset_0725.xlsx
# handbook, used to leave the target equation's exact form out of GPT
# pretraining (see kd/model/eqgpt/vocab.py:read_dataset).
_KNOWN_EQUATION_NAMES = {
    "kdv": "KdV equation",
}


class KD_EqGPT(BaseEstimator):
    """
    Generative PDE-structure discovery baseline (EqGPT), wrapping the
    original repo's first three pipeline stages: GPT pretraining on a
    221-equation math-handbook corpus, a surrogate neural network fit to
    (possibly sparse/noisy) solution data, and a reward-guided search that
    samples PDE-term sequences from the GPT and fine-tunes it on the
    best-scoring candidates found so far.

    Not covered: the original repo's fourth stage, PINN-based coefficient
    refinement of the discovered structure.

    Follows the scikit-learn-style API used by other `kd` baselines
    (see `kd.model.kd_sga.KD_SGA`): configure via `__init__`, run via
    `fit`, read results from `self.best_pde_` / `self.best_award_`.
    """

    def __init__(
        self,
        gpt_pretrain_epochs: int = 20,
        gpt_batch_size: int = 128,
        augment_times: int = 32,
        surrogate_iters: int = 5000,
        surrogate_neurons: int = 50,
        surrogate_layers: int = 5,
        activation: str = "Sin",
        choose: int = 2000,
        choose_validate: int = 500,
        noise_level: float = 0.0,
        optimize_epochs: int = 3,
        samples: int = 100,
        epi: float = 0.2,
        meta_nx: int = 60,
        meta_nt: int = 60,
        margin_frac: float = 0.1,
        seed: int = 0,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        gpt_pretrain_epochs, gpt_batch_size, augment_times :
            Supervised pretraining settings for the GPT on the handbook
            corpus (excluding the target equation).
        surrogate_iters, surrogate_neurons, surrogate_layers, activation :
            Surrogate network (fits noisy/sparse u(x,t)) training settings.
            `activation` is one of "Sin"/"Tanh"/"Rational".
        choose, choose_validate :
            Number of (x,t) samples used to train / validate the surrogate.
        noise_level :
            Percentage (0-100) of Gaussian noise (relative to std(u)) added
            to the training data, matching the original repo's noise model.
        optimize_epochs, samples :
            Reward-guided search: number of rounds, and candidate sequences
            sampled from the GPT per round.
        epi :
            Sparsity penalty coefficient in the reward
            `(1 - epi*log10(n_terms)) * R^2`.
        meta_nx, meta_nt :
            Resolution of the dense grid used to evaluate surrogate
            derivatives (the "meta-data") for reward computation.
        margin_frac :
            Fraction of the dataset's x/t range trimmed from each edge
            when building the meta-data grid, to avoid boundary artifacts
            in the surrogate's derivative estimates.
        seed : Random seed (numpy + torch + python `random`).
        device : "cuda"/"cpu"/None (auto-detect).
        """
        self.gpt_pretrain_epochs = gpt_pretrain_epochs
        self.gpt_batch_size = gpt_batch_size
        self.augment_times = augment_times
        self.surrogate_iters = surrogate_iters
        self.surrogate_neurons = surrogate_neurons
        self.surrogate_layers = surrogate_layers
        self.activation = activation
        self.choose = choose
        self.choose_validate = choose_validate
        self.noise_level = noise_level
        self.optimize_epochs = optimize_epochs
        self.samples = samples
        self.epi = epi
        self.meta_nx = meta_nx
        self.meta_nt = meta_nt
        self.margin_frac = margin_frac
        self.seed = seed
        self.device = device
        self.verbose = verbose

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        dataset: Any,
        equation_name: str = "kdv",
        variables: Sequence[str] = ("x", "t"),
    ) -> "KD_EqGPT":
        """
        Discover a PDE structure for `dataset` (a `kd.dataset.GridPDEDataset`,
        e.g. from `kd.dataset.load_kdv_equation()`).

        Parameters
        ----------
        dataset : GridPDEDataset
            Must expose `.x`, `.t`, `.usol` (legacy 1D-spatial layout).
        equation_name : str
            Short key used to look up the exact handbook row to exclude
            from GPT pretraining (see `_KNOWN_EQUATION_NAMES`); unknown
            keys train on the full handbook (no held-out row).
        variables : sequence of str
            Input variable names, in the order matching `dataset`'s (x, t)
            columns. Only ("x", "t") -- the 1D time-dependent case -- has
            been exercised.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        device = self._resolve_device()
        set_device(device)

        data = dataset.get_data()
        x = np.asarray(data["x"], dtype=float).reshape(-1)
        t = np.asarray(data["t"], dtype=float).reshape(-1)
        usol = np.real(np.asarray(data["usol"], dtype=float))
        if usol.ndim == 3:
            usol = usol[0]
        if usol.shape != (len(x), len(t)):
            usol = usol.T
        if usol.shape != (len(x), len(t)):
            raise ValueError(f"usol shape {usol.shape} incompatible with x({len(x)}), t({len(t)})")

        if self.noise_level > 0:
            noise = (self.noise_level / 100.0) * np.std(usol) * np.random.randn(*usol.shape)
            usol = usol + noise

        # Pretrain the GPT while leaving out the target equation.
        exclude_name = _KNOWN_EQUATION_NAMES.get(equation_name.lower(), "")
        train_sequences = get_train_dataset(exclude_name, self.augment_times, seed=self.seed)
        model_q = GPT()
        pretrain_gpt(
            model_q,
            train_sequences,
            device,
            epochs=self.gpt_pretrain_epochs,
            batch_size=self.gpt_batch_size,
            verbose=self.verbose,
        )

        # Fit the surrogate network to the sampled solution field.
        net = NN(
            num_hidden_layers=self.surrogate_layers,
            neurons_per_layer=self.surrogate_neurons,
            input_dim=2,
            output_dim=1,
            activation=self.activation,
        )
        h_train, h_val, x_train, x_val = random_data(
            x,
            t,
            usol,
            choose=min(self.choose, len(x) * len(t) - self.choose_validate),
            choose_validate=self.choose_validate,
            seed=self.seed,
        )
        net = train_surrogate_model(
            net,
            x_train,
            h_train,
            x_val,
            h_val,
            device,
            n_iter=self.surrogate_iters,
            verbose=self.verbose,
        )

        # Search for structures using the regression reward.
        margin_x = self.margin_frac * (x.max() - x.min())
        margin_t = self.margin_frac * (t.max() - t.min())
        database = generate_meta_grid(
            x.min() + margin_x,
            x.max() - margin_x,
            t.min() + margin_t,
            t.max() - margin_t,
            nx=self.meta_nx,
            nt=self.meta_nt,
            device=device,
        )

        mask_invalid = get_mask_invalid(variables, device)
        words2value: Dict[str, np.ndarray] = {}

        best_sentences: list = []
        best_awards: list = []
        history = []

        for epoch in range(self.optimize_epochs):
            all_reward = torch.zeros(self.samples)
            all_sentences = []
            for i in range(self.samples):
                sentence = [word2id["S"]]
                while len(sentence) < 49:  # MAX_POS - 1
                    next_token, _ = model_q.step(sentence, mask_invalid)
                    sentence.append(int(next_token))
                    if next_token == word2id["E"]:
                        break
                sentence.pop(0)  # drop the start token before scoring

                reward, words2value, mask_invalid = calculate_reward(
                    sentence, net, database, words2value, mask_invalid, variables, self.epi
                )
                all_reward[i] = reward
                all_sentences.append(sentence)

            top_index, top_award = find_min_no_repeat(all_reward, k=10)
            if epoch == 0:
                best_sentences = [all_sentences[i] for i in top_index]
                best_awards = list(top_award)
            else:
                for idx, award in zip(top_index, top_award):
                    if award in best_awards:
                        continue
                    worse = [j for j, a in enumerate(best_awards) if a < award]
                    if not worse:
                        continue
                    insert_at = worse[0]
                    best_sentences.insert(insert_at, all_sentences[idx])
                    best_awards.insert(insert_at, award)
                    best_sentences.pop(-1)
                    best_awards.pop(-1)

            history.append(
                {
                    "epoch": epoch,
                    "best_award": best_awards[0] if best_awards else 0.0,
                    "best_equation": sentence_to_str(best_sentences[0]) if best_sentences else "",
                }
            )
            if self.verbose:
                print(
                    f"[EqGPT search] epoch {epoch + 1}/{self.optimize_epochs} "
                    f"best={history[-1]['best_equation']} award={history[-1]['best_award']:.4f}"
                )

            # Fine-tune the GPT on the current elite set (this actually
            # updates model_q, unlike the original repo's optimize loop,
            # which -- apparently by mistake -- trained a freshly
            # re-initialized GPT each epoch that was never used for
            # subsequent sampling; see continue_train_GPT.py's `__main__`).
            elite_sequences = []
            for sentence in best_sentences:
                seq = [word2id["S"]] + list(sentence)
                if seq[-1] != word2id["E"]:
                    seq.append(word2id["E"])
                elite_sequences.append(seq)
            pretrain_gpt(
                model_q,
                elite_sequences,
                device,
                epochs=3,
                batch_size=len(elite_sequences),
                lr=1e-5,
                verbose=False,
            )

        self.best_pde_ = history[-1]["best_equation"] if history else ""
        self.best_award_ = history[-1]["best_award"] if history else 0.0
        self.history_ = history
        self.model_ = model_q
        self.surrogate_ = net
        self.dataset_ = dataset

        return self
