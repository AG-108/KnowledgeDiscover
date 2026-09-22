from _common import bootstrap_project_root

project_root = bootstrap_project_root()

from kd.model.kd_sga import KD_SGA

# Configure the parameters supported by SGA's SolverConfig.
model = KD_SGA(sga_run=10, depth=3)

# Select the built-in problem and train the solver.
model.fit(problem_name="chafee-infante")

# Inspect the discovered equation and score.
print(f"The discovered equation is: {model.best_pde_}")

# Render the solver diagnostics.
model.plot_results()
