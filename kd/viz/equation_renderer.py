# equation_renderer.py

import matplotlib.pyplot as plt


def render_latex_to_image(
    latex_str,
    output_path=None,
    font_size=16,
    dpi=300,
    background_color="white",
    base_figsize_width=8,
    base_figsize_height=2,  # Use these dimensions as lower bounds for dynamic figure sizing.
):
    try:
        # Increase figure width with expression length to reduce clipping.
        # Raw string length is a sufficient approximation for layout purposes.
        estimated_chars = len(latex_str)
        # Add one inch per ten characters and cap the width at 30 inches.
        adjusted_width = base_figsize_width + (max(0, estimated_chars - 50) / 15.0)
        adjusted_width = max(base_figsize_width, min(adjusted_width, 40))

        fig, ax = plt.subplots(
            figsize=(adjusted_width, base_figsize_height), facecolor=background_color
        )

        # Additional text properties can be supplied here if needed.
        ax.text(
            0.5, 0.5, latex_str, size=font_size, ha="center", va="center", color="black", wrap=True
        )
        ax.axis("off")  # Hide axes and borders around the rendered equation.

        if output_path is not None:
            plt.savefig(
                output_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.1,
                facecolor=background_color,
            )
            print(f"[Equation Renderer INFO] 图像已成功保存到: {output_path}")

        plt.show()

    except Exception as e:
        print(f"[Equation Renderer Error] 渲染 LaTeX 到图像时发生错误: {e}")
        print(f"  错误的 LaTeX 内容可能是: {latex_str}")
    finally:
        if "fig" in locals():  # Close the figure only if allocation succeeded.
            plt.close(fig)
