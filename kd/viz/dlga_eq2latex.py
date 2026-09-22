# dlga_eq2latex.py

from collections import Counter  # Count repeated genes when rendering powers.

import numpy as np  # Coefficients are represented as NumPy arrays.

# Map DLGA gene indices to operator names.
DLGA_INTERNAL_TERM_NAMES = ["u", "ux", "uxx", "uxxx", "ut", "utt"]

# Allow callers to override the default LaTeX style map.
DEFAULT_LATEX_STYLE_MAP = {
    "u": "u",
    "ux": "u_x",
    "uxx": "u_{xx}",
    "uxxx": "u_{xxx}",
    "ut": "u_t",  # A full partial-derivative form can be supplied through the style map.
    "utt": "u_{tt}",  # r"\frac{\partial^2 u}{\partial t^2}"
}


def _module_to_latex_base_term(module_indices, dlga_term_names, latex_style_map):
    """Convert one gene-index module to a coefficient-free LaTeX term."""
    if not module_indices:  # Treat an empty module as the multiplicative identity.
        return "1"  # Its fitted coefficient therefore becomes a constant term.

    gene_counts = Counter(module_indices)
    latex_parts = []

    # Sort genes so equivalent products always render in the same order.
    for gene_idx in sorted(gene_counts.keys()):
        count = gene_counts[gene_idx]
        term_name_dlga = (
            dlga_term_names[gene_idx]
            if 0 <= gene_idx < len(dlga_term_names)
            else f"未知G({gene_idx})"
        )
        term_latex_base = latex_style_map.get(term_name_dlga, term_name_dlga)

        if count > 1:  # Render repeated genes as powers.
            latex_parts.append(f"{term_latex_base}^{{{count}}}")
        else:
            latex_parts.append(term_latex_base)

    return " ".join(latex_parts)
    # For example, ['u', 'u_x^{2}', 'u_{xxx}^{2}'] is rendered as a product.
    # The resulting term is 'u u_x^{2} u_{xxx}^{2}'.


def _format_full_latex_term(coeff_val, base_term_latex, is_first_rhs_term):
    """Format one signed coefficient and its LaTeX base term."""
    # Format coefficient magnitudes to four decimal places.
    coeff_abs_str = f"{abs(coeff_val):.4f}"  # Handle the sign separately from the magnitude.

    # Normalize negative zero to '0.0000'.
    if np.isclose(coeff_val, 0.0):
        sign = ""  # Handle sign placement after building the coefficient text.
        coeff_display = "0.0000"
    elif coeff_val > 0:
        sign = "+"
        coeff_display = coeff_abs_str
    else:  # coeff_val < 0
        sign = "-"
        coeff_display = coeff_abs_str

    # Omit a unit coefficient for nonconstant symbolic terms.
    if (
        np.isclose(abs(coeff_val), 1.0) and base_term_latex != "1"
    ):  # Preserve unit coefficients when the base term is a constant.
        formatted_term = base_term_latex
    elif base_term_latex == "1":  # Render a constant module as its coefficient alone.
        formatted_term = coeff_display
    else:  # Combine general coefficients and base expressions with one space.
        formatted_term = f"{coeff_display} {base_term_latex}"

    # Add the leading operator according to sign and term position.
    if is_first_rhs_term:
        if sign == "-":
            return f"- {formatted_term}"
        else:  # A nonnegative first term has no leading plus sign.
            return formatted_term
    else:  # Later terms always include an explicit operator.
        if sign == "-":
            return f"- {formatted_term}"  # coeff=-0.5, base="u_x", is_first=False => "- 0.5000 u_x"
        else:  # Prefix later nonnegative terms with a plus sign.
            return f"+ {formatted_term}"


def chromosome_to_latex(
    chromosome,
    coefficients,
    lhs_name_str,
    dlga_term_names=DLGA_INTERNAL_TERM_NAMES,
    latex_style_map=DEFAULT_LATEX_STYLE_MAP,
):
    """Convert a DLGA chromosome and coefficients to a complete LaTeX equation."""
    # Convert the left-hand-side name through the style map.
    lhs_latex = latex_style_map.get(lhs_name_str, lhs_name_str)

    # Empty chromosomes or coefficient arrays represent a zero right-hand side.
    if not chromosome or coefficients is None or coefficients.size == 0:
        return f"${lhs_latex} = 0$"

    rhs_latex_full_terms = []
    processed_terms_count = 0  # Track the first retained right-hand-side term for sign formatting.

    # Pair each chromosome module with its coefficient.
    for i, module_indices in enumerate(chromosome):
        if i >= len(
            coefficients
        ):  # Stop if a malformed result omits coefficients for later modules.
            print(f"[警告] 模块索引 {i} 超出系数数组边界。")
            continue

        # Coefficients normally have shape (n_terms, 1).
        # Extract the scalar from the current coefficient row.
        coeff_val = coefficients[i, 0]

        # A caller may choose to suppress near-zero coefficients here.
        # if np.isclose(coeff_val, 0.0, atol=1e-7):
        #     continue
        # The current renderer includes every fitted term.

        # Render the base term with powers but without its coefficient.
        base_term = _module_to_latex_base_term(module_indices, dlga_term_names, latex_style_map)

        # Add the coefficient and sign to the base term.
        is_first = processed_terms_count == 0
        full_term_latex = _format_full_latex_term(coeff_val, base_term, is_first)

        # Mark subsequent terms so they receive an explicit operator.
        rhs_latex_full_terms.append(full_term_latex)
        processed_terms_count += 1

    # Join the rendered right-hand-side terms with spaces.
    final_rhs_latex = " ".join(rhs_latex_full_terms)
    if not final_rhs_latex:  # Fall back to zero if all terms were omitted.
        final_rhs_latex = "0"

    # Normalize any redundant leading plus sign defensively.
    # The formatter normally prevents a redundant leading plus sign.
    # Collapse duplicate spaces introduced by custom style values.
    final_rhs_latex = final_rhs_latex.replace("  ", " ")

    return f"${lhs_latex} = {final_rhs_latex}$"


def dlga_eq2latex(
    chromosome: list,
    coefficients: np.ndarray,
    lhs_name_str: str,
    operator_names: list[str] = None,  # Keep the operator mapping optional for compatibility.
) -> str:
    """
    Receives and prints the core equation data passed from DLGA.
    This is the first step, used to verify the data flow.
    """
    print("\n[Equation Renderer INFO] Successfully received data from DLGA:")

    if operator_names is not None:
        name = operator_names
        note = f"(Note: Using dynamic operator list for translation: {name})"
    else:
        name = DLGA_INTERNAL_TERM_NAMES
        note = f"(Note: Using default, hardcoded operator list for translation: {name})"
    print("\n[Equation Renderer INFO] " + note)

    print("  Original chromosome (list of modules, a module is a list of gene indices):")
    for i, module in enumerate(chromosome):
        print(f"    Term {i+1} - Module (gene indices): {module}")
        # Further explain the meaning of the module
        term_parts = [
            (
                DLGA_INTERNAL_TERM_NAMES[idx]
                if 0 <= idx < len(DLGA_INTERNAL_TERM_NAMES)
                else f"Unknown Gene({idx})"
            )
            for idx in module
        ]
        print(
            f"      Meaning (product of terms): {' * '.join(term_parts) if term_parts else 'Base constant term'}"
        )

    print("  Coefficients (NumPy array, corresponding to each module/term in the chromosome):")
    for i, coeff_row in enumerate(coefficients):  # coefficients is an (N, 1) array
        print(f"    Term {i+1} - Coefficient: {coeff_row[0]:.4f}")

    print(f"  Left-hand side term name (string): {lhs_name_str}")
    print(
        f"  (Note: Gene indices in the chromosome map to the following list: {DLGA_INTERNAL_TERM_NAMES})"
    )
    print("--- Equation Renderer data reception end ---\n")

    # Generate LaTeX string
    dlga_latex_str = chromosome_to_latex(
        chromosome=chromosome,
        coefficients=coefficients,
        lhs_name_str=lhs_name_str,
        # dlga_term_names and latex_style_map will use the function's default values
    )

    print(f"[Equation Renderer INFO]: LaTeX: {dlga_latex_str}")

    return dlga_latex_str
