from _common import bootstrap_project_root

project_root = bootstrap_project_root()


# Import project dependencies after bootstrapping the repository path.
from kd.dataset import load_kdv_equation
from kd.model.kd_dlga import KD_DLGA
from kd.viz.dlga_viz import (
    plot_derivative_relationships,
    plot_optimization_analysis,
    plot_pde_comparison,
    plot_pde_parity,
    plot_residual_analysis,
    plot_time_slices,
    plot_training_loss,
    plot_validation_loss,
)
from kd.viz.equation_renderer import render_latex_to_image

# Load the PDE field and sample training coordinates.
kdv_data = load_kdv_equation()

# For custom MAT files, set x_key, t_key, and u_key to match the stored arrays.
# kdv_data = load_pde_dataset(filename="KdV_equation.mat", x_key='x', t_key='tt', u_key='uu')

# Sample 1,000 points from the full field for training.
X_train, y_train = kdv_data.sample(n_samples=1000)


# Configure the DLGA search and its neural surrogate.
model = KD_DLGA(
    operators=[
        "u",
        "u_x",
        "u_xx",
        "u_xxx",
    ],  # Define the candidate operator library.
    epi=0.1,  # Penalize unnecessarily complex equations.
    input_dim=2,  # Match the input dimension to the number of coordinate columns.
    verbose=False,  # Print the best candidate from each generation.
    max_iter=9000,  # Limit the neural surrogate training iterations.
)


# Fit the model and predict the field on the full grid.
print("\nTraining DLGA model...")
model.fit(X_train, y_train)

print("\nGenerating predictions...")
X_full = kdv_data.mesh()  # Build the full coordinate grid used by the visualizations.
u_pred = model.predict(X_full)
u_pred = u_pred.reshape(kdv_data.get_size())


# Render the discovered equation and diagnostics.
render_latex_to_image(model.eq_latex)  # Render the best discovered equation.
plot_training_loss(model)  # Plot the training loss.
plot_validation_loss(model)  # Plot the validation loss.

# Plot fitness and complexity over generations.
plot_optimization_analysis(model)

# Compare the reference and predicted fields as heatmaps.
plot_pde_comparison(kdv_data.x, kdv_data.t, kdv_data.usol, u_pred)

# Plot pointwise residuals and their distribution over the domain.
plot_residual_analysis(model, X_train, y_train, kdv_data.usol, u_pred)

# Compare field slices at selected times.
plot_time_slices(kdv_data.x, kdv_data.t, kdv_data.usol, u_pred, slice_times=[0.25, 0.5, 0.75])

# Relate the strongest right-hand-side terms to the left-hand side.
plot_derivative_relationships(model)

# Generate a parity plot for the final equation.
plot_pde_parity(model, title="Final Validation of Discovered Equation")
