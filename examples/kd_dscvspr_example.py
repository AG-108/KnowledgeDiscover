import os

import numpy as np
from _common import bootstrap_project_root

project_root = bootstrap_project_root()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="numpy.*")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.*")


from kd.dataset import load_burgers_equation
from kd.model import KD_DSCV_SPR
from kd.viz.discover_eq2latex import discover_program_to_latex
from kd.viz.dscv_viz import (
    plot_density,
    plot_evolution,
    plot_expression_tree,
    plot_spr_actual_vs_predicted,
    plot_spr_field_comparison,
    plot_spr_residual_analysis,
)
from kd.viz.equation_renderer import render_latex_to_image

np.random.seed(42)

burgers_data = load_burgers_equation()

# burgers_data = load_pde_dataset(filename="KdV_equation.mat", x_key='x', t_key='tt', u_key='uu')

x, y = burgers_data.sample(n_samples=0.1)
lb, ub = burgers_data.mesh_bounds()

# PINN operators must be PyTorch-compatible and usually end in ``_t``.
model = KD_DSCV_SPR(
    n_samples_per_batch=100,  # Number of generated traversals by agent per batch
    binary_operators=["add_t", "mul_t", "div_t", "diff_t", "diff2_t"],
    unary_operators=["n2_t"],
)


step_output = model.fit(x, y, [lb, ub], n_epochs=10)

print(
    f"Current best expression is {step_output['expression']} and its reward is {step_output['r']}"
)

# Convert the best symbolic program to LaTeX and render it.
render_latex_to_image(discover_program_to_latex(step_output["program"]))

# Visualize the final equation as an expression tree.
plot_expression_tree(model)

# Plot the reward density to inspect candidate quality.
plot_density(model)

# Plot the best reward over training iterations.
plot_evolution(model)

# Analyze the residual of the discovered PDE.
plot_spr_residual_analysis(model, step_output["program"])

# Compare the PINN field with the reference field.
plot_spr_field_comparison(model, step_output["program"])

# Compare PINN predictions and targets in a parity plot.
plot_spr_actual_vs_predicted(model, step_output["program"])
