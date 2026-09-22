import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Public plot helpers control whether figures are displayed or saved.


def plot_expression_tree(model, output_dir: str = None):
    """Display the best program as an expression tree."""

    graph = model.searcher.plotter.tree_plot(model.searcher.best_p)

    if output_dir is None:
        png_bytes = graph.pipe(format="png")
        image = Image.open(io.BytesIO(png_bytes))
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        plt.close()
    else:
        pass


def plot_density(model, epoches=None, output_dir: str = None):
    """Plot the reward density for the requested training epochs."""

    model.plot(fig_type="density", epoches=epoches)


# Configure fonts that support English and Chinese labels.
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS", "Arial"]
plt.rcParams["axes.unicode_minus"] = False  # Preserve minus signs with fallback fonts.

PLOT_STYLE = {
    "font.size": 12,
    "figure.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.prop_cycle": plt.cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.sans-serif": ["SimHei", "DejaVu Sans", "Arial Unicode MS", "Arial"],
    "axes.unicode_minus": False,
}


def plot_evolution(model, figsize=(10, 6)):
    """
    Plot reward changes during training.

    Parameters:
        model: Trained KD_DSCV model.
        figsize (tuple, optional): Figure size.

    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        # Extract the reward history from the trained model.
        rewards = model.searcher.r_train

        # Derive the maximum, mean, and running-best reward series.
        x = np.arange(1, len(rewards) + 1)
        r_max = [np.max(r) for r in rewards]
        r_avg = [np.mean(r) for r in rewards]
        r_best = np.maximum.accumulate(np.array(r_max))

        # Plot all reward summaries on the same axes.
        ax.plot(x, r_best, linestyle="-", color="black", linewidth=2, label="Best Reward")
        ax.plot(x, r_max, color="#F47E62", linestyle="-.", linewidth=2, label="Maximum Reward")
        ax.plot(x, r_avg, color="#4F8FBA", linestyle="--", linewidth=2, label="Average Reward")

        ax.set_xlabel("Iteration Count")
        ax.set_ylabel("Reward Value")
        ax.grids(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=False)

        plt.show()

        plt.close()


# Internal numerical helpers.
def _finite_difference(y, x, order=1):
    """Approximate a derivative with centered finite differences."""
    if len(y.shape) > 1:
        y = y.flatten()
    if len(x.shape) > 1:
        x = x.flatten()
    dx = x[1] - x[0]
    if order == 1:
        return np.gradient(y, dx, edge_order=2)
    elif order == 2:
        return np.gradient(np.gradient(y, dx, edge_order=2), dx, edge_order=2)
    elif order == 3:
        return np.gradient(
            np.gradient(np.gradient(y, dx, edge_order=2), dx, edge_order=2), dx, edge_order=2
        )
    elif order == 4:
        return np.gradient(
            np.gradient(
                np.gradient(np.gradient(y, dx, edge_order=2), dx, edge_order=2), dx, edge_order=2
            ),
            dx,
            edge_order=2,
        )
    else:
        raise ValueError("只支持1-4阶导数, 但请求了 {order} 阶")


def _evaluate_term_recursively(node, u_snapshot, x_coords):
    """Evaluate a symbolic expression tree over the supplied field values."""
    if not node.children:
        if node.val == "u1":
            return u_snapshot
        elif node.val == "x1":
            return x_coords
        try:  # Numeric leaf nodes represent fitted constants.
            return float(node.val)
        except ValueError:
            raise ValueError(f"未知的叶子节点或无法转换为浮点数的常数: {node.val}")

    # Evaluate child nodes before applying the current operator.
    child_values = [
        _evaluate_term_recursively(child, u_snapshot, x_coords) for child in node.children
    ]
    op_name = node.val.removesuffix(
        "_t"
    )  # Normalize tensor-compatible operator suffixes before dispatch.

    if op_name == "add":
        return child_values[0] + child_values[1]
    elif op_name == "sub":
        return child_values[0] - child_values[1]
    elif op_name == "mul":
        return child_values[0] * child_values[1]
    elif op_name == "div":
        return child_values[0] / (child_values[1] + 1e-8)

    elif op_name == "n2":
        return child_values[0] ** 2
    elif op_name == "n3":
        return child_values[0] ** 3
    elif op_name == "n4":
        return child_values[0] ** 4
    elif op_name == "n5":
        return child_values[0] ** 5

    elif op_name == "inv":
        return 1.0 / (child_values[0] + 1e-8)
    elif op_name == "neg":
        return -child_values[0]

    elif op_name == "sin":
        return np.sin(child_values[0])
    elif op_name == "cos":
        return np.cos(child_values[0])
    elif op_name == "tan":
        return np.tan(child_values[0])

    elif op_name == "diff":
        return _finite_difference(child_values[0], x_coords, order=1)
    elif op_name == "diff2":
        return _finite_difference(child_values[0], x_coords, order=2)
    elif op_name == "diff3":
        return _finite_difference(child_values[0], x_coords, order=3)
    elif op_name == "diff4":
        return _finite_difference(child_values[0], x_coords, order=4)
    else:
        raise ValueError(f"未知的操作: {op_name}")


def _calculate_pde_fields(model, best_program):
    """Compute aligned reference, prediction, residual, and coordinate arrays."""
    # Read the fitted terms, coefficients, and reference field.
    data_dict = model.data_class.get_data()
    final_symbolic_terms = best_program.STRidge.terms
    w_best = best_program.w
    u_trimmed = data_dict["u"]
    ut_trimmed = data_dict["ut"]
    x_axis = data_dict["X"][0].flatten()

    # Evaluate each symbolic term over the full field.
    num_space_points, num_timesteps_trimmed = u_trimmed.shape
    Theta_final = np.zeros((u_trimmed.size, len(final_symbolic_terms)))
    for i, term_node in enumerate(final_symbolic_terms):
        term_values_grid = np.zeros_like(u_trimmed)
        for t_idx in range(num_timesteps_trimmed):
            u_snapshot = u_trimmed[:, t_idx]
            term_values_grid[:, t_idx] = _evaluate_term_recursively(term_node, u_snapshot, x_axis)
        Theta_final[:, i] = term_values_grid.flatten()

    if not np.isfinite(Theta_final).all():
        print("\n[计算警告]: 发现的方程包含数值不稳定的项")
        print("  后续的可视化可能会失败或显示不正确")
        # return None

    # Reconstruct the right-hand side and physical residual.
    if Theta_final.shape[1] == len(w_best) - 1:
        y_hat_rhs = Theta_final @ w_best[:-1] + w_best[-1]
    else:
        y_hat_rhs = Theta_final @ w_best
    physical_residual = ut_trimmed.flatten() - y_hat_rhs.flatten()

    # Build coordinates aligned with the flattened field arrays.
    t_axis_trimmed = np.arange(num_timesteps_trimmed)
    X_grid, T_grid = np.meshgrid(x_axis, t_axis_trimmed, indexing="ij")
    coords_for_plot = np.stack([X_grid.flatten(), T_grid.flatten()], axis=1)

    # Return aligned arrays for all public plotting functions.
    return {
        "residual": physical_residual,
        "coords": coords_for_plot,
        "ut_grid": ut_trimmed,
        "y_hat_grid": y_hat_rhs.reshape(num_space_points, num_timesteps_trimmed),
        "x_axis": x_axis,
        "t_axis": t_axis_trimmed,
    }


def plot_pde_residual_analysis(model, best_program, show_plot=True):
    """Plot the physical residual of a fitted DISCOVER program."""
    # Share the same aligned field calculation across plots.
    fields = _calculate_pde_fields(model, best_program)

    if show_plot:
        physical_residual = fields["residual"]
        coords_for_plot = fields["coords"]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sc = plt.scatter(
            coords_for_plot[:, 1],
            coords_for_plot[:, 0],
            c=physical_residual,
            cmap="coolwarm",
            s=15,
            alpha=0.8,
        )
        plt.colorbar(sc, label="Physical Residual ($u_t$ - RHS)")
        plt.xlabel("Time (trimmed, index)")
        plt.ylabel("Space")
        plt.title("Spatiotemporal Distribution of Residuals")
        plt.subplot(1, 2, 2)
        plt.hist(physical_residual, bins=50, density=True, edgecolor="black", alpha=0.7)
        plt.xlabel("Residual Value")
        plt.ylabel("Probability Density")
        plt.title("Residual Distribution")
        plt.tight_layout()
        plt.show()
    else:
        return fields["residual"], fields["coords"]


def plot_field_comparison(model, best_program, show_plot=True):
    """Compare the reconstructed and reference PDE fields."""
    # Reuse the common aligned field calculation.
    fields = _calculate_pde_fields(model, best_program)

    ut_grid = fields["ut_grid"]
    y_hat_grid = fields["y_hat_grid"]
    x_axis = fields["x_axis"]
    t_axis = fields["t_axis"]

    if show_plot:
        # Use a shared color range for a meaningful field comparison.
        vmin = min(ut_grid.min(), y_hat_grid.min())
        vmax = max(ut_grid.max(), y_hat_grid.max())

        # Place reference and predicted fields side by side.
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle("Predicted Field vs. True Field Comparison", fontsize=16)

        # Plot the reference field on the left.
        ax0 = axes[0]
        im0 = ax0.pcolormesh(
            t_axis, x_axis, ut_grid, cmap="viridis", vmin=vmin, vmax=vmax, shading="gouraud"
        )
        fig.colorbar(im0, ax=ax0, label="Value")
        ax0.set_title("True Field ($u_t$)", fontsize=14)
        ax0.set_xlabel("Time (trimmed, index)", fontsize=12)
        ax0.set_ylabel("Space", fontsize=12)

        # Plot the reconstructed field on the right.
        ax1 = axes[1]
        im1 = ax1.pcolormesh(
            t_axis, x_axis, y_hat_grid, cmap="viridis", vmin=vmin, vmax=vmax, shading="gouraud"
        )
        fig.colorbar(im1, ax=ax1, label="Value")
        ax1.set_title("Predicted Field (RHS)", fontsize=14)
        ax1.set_xlabel("Time (trimmed, index)", fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        return ut_grid, y_hat_grid, x_axis, t_axis


def plot_actual_vs_predicted(model, best_program):
    """
    Plots an "Actual vs. Predicted" scatter plot with a 45-degree reference line.

    Args:
        model: The trained KD_DSCV model instance.
        best_program: The final Program object discovered by the model.
    """
    print("Generating 'Actual vs. Predicted' plot...")
    # Call the helper function to get all computed fields
    fields = _calculate_pde_fields(model, best_program)

    y_true = fields["ut_grid"]
    y_pred = fields["y_hat_grid"]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, label="Predicted Points")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = 0.1 * (max_val - min_val) if (max_val - min_val) > 0 else 0.1
    plot_limit = (min_val - margin, max_val + margin)

    plt.plot(plot_limit, plot_limit, "r--", label="Perfect Prediction (y=x)")

    plt.xlabel("True Values (Ground Truth, $u_t$)")
    plt.ylabel("Predicted Values (RHS)")
    plt.title("Actual vs. Predicted")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.xlim(plot_limit)
    plt.ylim(plot_limit)
    plt.show()


def _calculate_pinn_fields(model, best_program):
    """Read aligned PINN field values cached by a fitted program."""
    # Trigger reward evaluation so the program caches Theta and predictions.
    _ = best_program.r_ridge

    # Read the aligned values produced during reward evaluation.
    y_hat_rhs = best_program.y_hat_rhs
    Theta = best_program.Theta

    if y_hat_rhs is None or Theta is None:
        raise AttributeError(
            "best_program 对象中未找到缓存的 Theta 矩阵或 y_hat_rhs。请确保对 pde_pinn.py 和 program.py 的修改已生效。"
        )

    # Read the cropped target arrays from the active task.
    task = best_program.task
    ut_full = task.ut
    x_coords_tensor = task.x[0]
    t_coords_tensor = task.t

    # Move task coordinates to NumPy for plotting.
    x_coords_np = x_coords_tensor.cpu().detach().numpy()
    t_coords_np = t_coords_tensor.cpu().detach().numpy()

    # Compute the physical residual on the sampled points.
    physical_residual = ut_full.flatten() - y_hat_rhs.flatten()

    # Combine spatial and time coordinates into an (N, 2) array.
    coords_for_plot = np.hstack([x_coords_np, t_coords_np])

    # Return aligned arrays for the PINN visualization functions.
    return {
        "residual": physical_residual,
        "coords": coords_for_plot,
        "coords_x": x_coords_np.flatten(),
        "coords_t": t_coords_np.flatten(),
        "y_true": ut_full.flatten(),
        "y_pred": y_hat_rhs.flatten(),
    }


def plot_spr_residual_analysis(model, best_program):
    """Plot residual diagnostics for sparse PINN samples."""

    # Use the PINN-specific cache and sampled coordinates.
    fields = _calculate_pinn_fields(model, best_program)

    physical_residual = fields["residual"]
    coords_for_plot = fields["coords"]  # Coordinates are aligned as (space, time) rows.

    # Fail early if cached values no longer match sampled coordinates.
    if coords_for_plot.shape[0] != len(physical_residual):
        raise ValueError(
            f"坐标点数量 ({coords_for_plot.shape[0]}) 与残差值数量 ({len(physical_residual)}) 不匹配。"
        )

    plt.figure(figsize=(12, 5))

    # Plot the residual over space and time on the left.
    plt.subplot(1, 2, 1)
    # Coordinate column 0 is space and column 1 is time.
    sc = plt.scatter(
        coords_for_plot[:, 1],
        coords_for_plot[:, 0],
        c=physical_residual,
        cmap="coolwarm",
        s=15,
        alpha=0.8,
    )
    plt.colorbar(sc, label="Physical Residual ($u_t$ - RHS)")
    plt.xlabel("Time")
    plt.ylabel("Space")
    plt.title("Spatiotemporal Distribution of Residuals (on meta-data points)")

    # Plot the residual distribution on the right.
    plt.subplot(1, 2, 2)
    plt.hist(physical_residual, bins=50, density=True, edgecolor="black", alpha=0.7)
    plt.xlabel("Residual Value")
    plt.ylabel("Probability Density")
    plt.title("Residual Distribution")

    plt.tight_layout()
    plt.show()


def plot_spr_actual_vs_predicted(model, best_program):
    """Plot PINN targets against reconstructed right-hand-side values."""
    print("Generating 'Actual vs. Predicted' plot for PINN model...")
    fields = _calculate_pinn_fields(model, best_program)

    y_true = fields["y_true"]
    y_pred = fields["y_pred"]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, label="Predicted Points")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = 0.1 * (max_val - min_val) if (max_val - min_val) > 0 else 0.1
    plot_limit = (min_val - margin, max_val + margin)

    plt.plot(plot_limit, plot_limit, "r--", label="Perfect Prediction (y=x)")

    plt.xlabel("True Values (Ground Truth, $u_t$)")
    plt.ylabel("Predicted Values (RHS)")
    plt.title("Actual vs. Predicted (for PINN model)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.xlim(plot_limit)
    plt.ylim(plot_limit)
    plt.show()


def plot_spr_field_comparison(model, best_program):
    """Compare reconstructed and reference fields from irregular PINN samples."""
    print("Generating field comparison plot for PINN model...")
    fields = _calculate_pinn_fields(model, best_program)

    x, t, ut, y_hat = fields["coords_x"], fields["coords_t"], fields["y_true"], fields["y_pred"]

    # Triangulate the irregular samples before rendering the field.
    from scipy.interpolate import griddata

    grid_x, grid_t = np.mgrid[min(x) : max(x) : 100j, min(t) : max(t) : 100j]

    grid_ut = griddata((x, t), ut, (grid_x, grid_t), method="cubic")
    grid_yhat = griddata((x, t), y_hat, (grid_x, grid_t), method="cubic")

    # Share a common color scale across both fields.
    vmin = np.nanmin([grid_ut, grid_yhat])
    vmax = np.nanmax([grid_ut, grid_yhat])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Field Comparison (on Sparse Points)", fontsize=16)

    # Plot the reference field on the left.
    ax0 = axes[0]
    im0 = ax0.imshow(
        grid_ut.T,
        extent=(min(t), max(t), min(x), max(x)),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im0, ax=ax0, label="Value")
    ax0.set_title("True Field ($u_t$)")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Space")

    # Plot the reconstructed field on the right.
    ax1 = axes[1]
    im1 = ax1.imshow(
        grid_yhat.T,
        extent=(min(t), max(t), min(x), max(x)),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im1, ax=ax1, label="Value")
    ax1.set_title("Predicted Field (RHS)")
    ax1.set_xlabel("Time")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
