import sympy
from sympy.parsing.sympy_parser import parse_expr

from kd.viz.dlga_eq2latex import _format_full_latex_term

# Known limitation: this is a fixed default symbol table (single state
# variable `u1`, up to 3 spatial dims `x1..x3`, plus `c`). Callers can
# override it per-call via `discover_program_to_latex(custom_deeprl_symbols=...)`,
# but if a Program uses more state/input variables than this default and no
# override is passed, sympy.parse_expr will fail to resolve the extra symbol
# names. Ideally this table would be derived automatically from
# `Program.library` instead of hardcoded here.
DEEPRL_SYMBOLS_FOR_SYMPY = {
    name: sympy.Symbol(name)
    for name in ["u1", "x1", "x2", "x3", "c"]  # 'c' denotes an optional symbolic constant.
}

DEBUG_RENDERER_MODE = True  # Enable verbose conversion diagnostics when requested.


# Node -> LaTeX
def _discover_term_node_to_latex(term_node_obj, local_sympy_symbols=None):
    if local_sympy_symbols is None:
        local_sympy_symbols = DEEPRL_SYMBOLS_FOR_SYMPY
    try:
        if not hasattr(
            term_node_obj, "to_sympy_string"
        ):  # Term nodes must expose the conversion method used by DISCOVER.
            raise AttributeError("Node 对象没有 to_sympy_string 方法")

        sympy_expr_str = term_node_obj.to_sympy_string()

        if DEBUG_RENDERER_MODE:
            print(
                f"[discover_eq2latex INFO] Node '{repr(term_node_obj)}'.to_sympy_string() -> '{sympy_expr_str}'"
            )

        parsed_sympy_expr = parse_expr(
            sympy_expr_str, local_dict=local_sympy_symbols, transformations="all"
        )  # The full transformation set accepts the wider syntax emitted by DISCOVER.
        if DEBUG_RENDERER_MODE:
            print(f"[discover_eq2latex INFO] 解析后的SymPy表达式: {parsed_sympy_expr}")

        latex_output = sympy.latex(parsed_sympy_expr)
        return latex_output
    except Exception as e:  # Preserve conversion failures in the rendered output.
        import traceback  # Include a traceback in debug mode.

        error_repr = repr(term_node_obj) if term_node_obj else "None"
        sympy_str_val = sympy_expr_str if "sympy_expr_str" in locals() else "未成功生成SymPy字符串"
        print(
            f"[ERROR] _discover_term_node_to_latex 中转换节点 '{error_repr}' (SymPy str: '{sympy_str_val}') 失败: {e}"
        )
        if DEBUG_RENDERER_MODE:
            traceback.print_exc()
        return f"\\text{{Error converting node: {error_repr}}}"


def discover_program_to_latex(
    program_object,  # lhs_name_str,
    # Override the module-level debug flag for one conversion.
    custom_lhs_latex_map=None,
    custom_deeprl_symbols=None,
):
    """Convert a fitted DISCOVER program to a complete LaTeX equation."""
    # Render the left-hand side.
    # current_lhs_map = custom_lhs_latex_map if custom_lhs_latex_map else DEFAULT_LATEX_STYLE_MAP
    # lhs_latex = current_lhs_map.get(lhs_name_str, lhs_name_str)
    lhs_latex = "u_t"  # DISCOVER currently uses u_t as the left-hand side.

    # Validate the fitted program structure before rendering terms.
    if not (
        program_object
        and hasattr(program_object, "w")
        and hasattr(program_object, "STRidge")
        and hasattr(program_object.STRidge, "terms")
        and program_object.w is not None
        and program_object.STRidge.terms is not None
        and len(program_object.w) == len(program_object.STRidge.terms)
    ):
        if DEBUG_RENDERER_MODE:
            w_status = str(getattr(program_object, "w", "未找到w属性"))
            terms_status = str(
                getattr(getattr(program_object, "STRidge", None), "terms", "未找到STRidge.terms")
            )
            print(
                "[渲染器警告] discover_program_to_latex: program_object 无效或缺少 w/STRidge.terms。"
            )
            print(f"  program_object: {program_object}")
            print(f"  w: {w_status}")
            print(f"  STRidge.terms: {terms_status}")
            if (
                hasattr(program_object, "w")
                and hasattr(program_object.STRidge, "terms")
                and program_object.w is not None
                and program_object.STRidge.terms is not None
            ):
                print(
                    f"  len(w)={len(program_object.w)}, len(STRidge.terms)={len(program_object.STRidge.terms)}"
                )
        return f"${lhs_latex} = 0 \\; (\\text{{Error: Invalid program structure}})$"

    coefficients = program_object.w
    term_nodes = program_object.STRidge.terms

    if not term_nodes:  # Use zero when the program contains no symbolic terms.
        return f"${lhs_latex} = 0$"

    # Render each right-hand-side term with its fitted coefficient.
    rhs_latex_full_terms = []
    processed_terms_count = 0  # Track the first retained term so signs are formatted correctly.

    current_sympy_symbols = (
        custom_deeprl_symbols if custom_deeprl_symbols else DEEPRL_SYMBOLS_FOR_SYMPY
    )

    for i, coeff_val_from_w in enumerate(coefficients):
        # program.w may be a Python list or a one-dimensional NumPy array.
        coeff_val = float(
            coeff_val_from_w
        )  # Convert scalar array values to ordinary Python floats.
        term_node = term_nodes[i]

        # Very small coefficients may be omitted from the rendered equation.
        # if np.isclose(coeff_val, 0.0, atol=1e-7):
        #     if DEBUG_RENDERER_MODE:
        # Debug output for omitted coefficients can be added here if needed.
        #     continue

        # Convert the symbolic node to a coefficient-free LaTeX term.
        base_term_latex = _discover_term_node_to_latex(
            term_node, local_sympy_symbols=current_sympy_symbols
        )

        # Keep conversion errors visible in the final equation.
        if DEBUG_RENDERER_MODE and "\\text{Error" in base_term_latex:
            print(
                f"[渲染器警告] 基础项 {repr(term_node)} 转换为LaTeX时出错，内容为: {base_term_latex}"
            )

        is_first = processed_terms_count == 0

        # Combine the coefficient, sign, and base term consistently.
        full_term_latex = _format_full_latex_term(coeff_val, base_term_latex, is_first)

        rhs_latex_full_terms.append(full_term_latex)
        processed_terms_count += 1

    final_rhs_latex = " ".join(rhs_latex_full_terms)
    if (
        not final_rhs_latex or processed_terms_count == 0
    ):  # Render zero if no right-hand-side term survives filtering.
        final_rhs_latex = "0"

    final_rhs_latex = final_rhs_latex.replace(
        "  ", " "
    )  # Normalize whitespace in the final expression.

    return f"${lhs_latex} = {final_rhs_latex}$"
