"""Conditional training IC and explicitly exploratory held-out complexity scores.

No source row is mutated. Scores are comparable only within the same observations,
target scale and residual domain. Adaptive structure search is not penalized.

Code evidence: run_benchmark.py regression_problem applies max_train_samples
before fit and records len(X_train) as n_train; run_regression passes exactly
these arrays to LLM-SR. kd/model/kd_llmsr.py _fit_program minimizes the ordinary
mean squared residual without clipping/normalization, using BFGS from ones.
fit stores best.mse in method_provenance.training_mse. _inline_body resolves
locals and _compile_expression allocates max(parameter index)+1 coordinates;
unused coordinates are excluded here. Functional redundancies are not resolved.
"""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cloud_ic_calculator", ROOT / "kd/evaluation/information_criteria.py")
_calculator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_calculator)


def _integer(value, minimum=1):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def parameter_count(program):
    """Mirror LLM-SR's inlining; count distinct used slots, including sparse indices.

    Unused optimizer coordinates are not identifiable fitted coefficients. This
    does not establish functional independence of the remaining coefficients.
    """
    function = ast.parse(program).body[0]
    if not isinstance(function, ast.FunctionDef):
        raise ValueError("Expected function")
    local = {}

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node):
            return copy.deepcopy(local[node.id]) if isinstance(node.ctx, ast.Load) and node.id in local else node

    def substitute(node):
        value = Substitute().visit(copy.deepcopy(node))
        if sum(1 for _ in ast.walk(value)) > 5000:
            raise ValueError("Expression too large")
        return value

    def bind(target, value):
        if isinstance(target, ast.Name):
            local[target.id] = value
            return
        if not isinstance(target, (ast.Tuple, ast.List)):
            raise ValueError("Unsupported assignment")
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and value.value.id in {"p", "params"} and isinstance(value.slice, ast.Slice):
            start = 0 if value.slice.lower is None else ast.literal_eval(value.slice.lower)
            stop = ast.literal_eval(value.slice.upper)
            if value.slice.step is not None or not _integer(start, 0) or not _integer(stop, 0):
                raise ValueError("Unsupported slice")
            values = [ast.Subscript(value=ast.Name(id="params", ctx=ast.Load()), slice=ast.Constant(i), ctx=ast.Load()) for i in range(start, stop)]
        elif isinstance(value, (ast.Tuple, ast.List)):
            values = value.elts
        else:
            raise ValueError("Unsupported tuple")
        if len(values) != len(target.elts):
            raise ValueError("Tuple size mismatch")
        for item, val in zip(target.elts, values):
            bind(item, val)

    for statement in function.body:
        if isinstance(statement, ast.Return):
            expression = substitute(statement.value)
            indices = set()
            parameter_names = {id(node.value) for node in ast.walk(expression)
                               if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                               and node.value.id in {"params", "p"}}
            for node in ast.walk(expression):
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in {"params", "p"}:
                    index = ast.literal_eval(node.slice)
                    if not _integer(index, 0):
                        raise ValueError("Nonliteral parameter index")
                    indices.add(index)
                elif isinstance(node, ast.Name) and re.fullmatch(r"p[0-9]+", node.id):
                    indices.add(int(node.id[1:]))
                elif isinstance(node, ast.Name) and node.id in {"params", "p"} and id(node) not in parameter_names:
                    raise ValueError("Bare parameter array has unknown parameter count")
            return len(indices)
        if isinstance(statement, ast.Assign):
            value = substitute(statement.value)
            for target in statement.targets:
                bind(target, value)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            bind(statement.target, substitute(statement.value))
        elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
            local[statement.target.id] = ast.BinOp(left=substitute(ast.Name(id=statement.target.id, ctx=ast.Load())), op=statement.op, right=substitute(statement.value))
        elif isinstance(statement, (ast.Pass, ast.Expr)):
            if isinstance(statement, ast.Expr) and not isinstance(statement.value, ast.Constant):
                raise ValueError("Unsupported expression statement")
        elif isinstance(statement, ast.Import) and all(a.name == "numpy" and a.asname in {None, "np"} for a in statement.names):
            continue
        else:
            raise ValueError("Unsupported statement")
    raise ValueError("Missing return")


def _score(n, mse, k):
    if not _integer(n) or not _integer(k, 0) or isinstance(mse, bool):
        raise ValueError("Unknown count or invalid residual")
    mse = float(mse)
    if not math.isfinite(mse) or mse <= 0:
        raise ValueError("Residual must be finite and positive; zero is degenerate")
    return _calculator.information_criteria(n_observations=n, n_free_parameters=k,
        residual_sum_squares=n * mse, likelihood_model="gaussian_mle_variance")


def training_ic(row):
    result = dict(aic=None, bic=None, training_ic_status="unavailable")
    if row.get("status") != "ok" or row.get("model") != "llmsr" or row.get("task") not in {"sr", "ode"}:
        return result
    try:
        k = parameter_count(row["best_program"])
        scores = _score(row["n_train"], row["method_provenance"]["training_mse"], k)
    except (KeyError, TypeError, ValueError, SyntaxError, RecursionError, OverflowError) as exc:
        result.update(training_ic_status="abstained", training_ic_reason=str(exc))
        return result
    result.update(scores)
    result.update(training_ic_status="conditional_training_plugin", training_ic_domain="training_observations",
        training_ic_parameter_source="distinct active params[i]/pi after AST local inlining",
        training_ic_reason="Unclipped training MSE; n_train after sampling. BFGS from ones is not a certified MLE; active coefficients may be redundant; adaptive search unpenalized.")
    return result


def proxy_ic(row):
    result = dict(aic_proxy_heldout_complexity=None, bic_proxy_heldout_complexity=None, proxy_ic_status="unavailable")
    if row.get("status") != "ok":
        return result
    pde = row.get("task") == "pde"
    n_key = "equation_residual_total_points" if pde else "n_test"
    mse_key = "equation_residual_mse" if pde else "test_mse"
    coverage_key = "equation_residual_finite_coverage" if pde else "prediction_finite_coverage"
    total_key = "equation_residual_total_points" if pde else "prediction_total_points"
    finite_key = "equation_residual_points" if pde else "prediction_finite_points"
    try:
        n = row[n_key]
        if row.get(coverage_key) != 1 or row.get(total_key) != n or row.get(finite_key) != n:
            raise ValueError("Full matching residual coverage required")
        scores = _score(n, row[mse_key], row["expression_complexity"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result.update(proxy_ic_status="abstained", proxy_ic_reason=str(exc))
        return result
    result.update(aic_proxy_heldout_complexity=scores["aic"], bic_proxy_heldout_complexity=scores["bic"],
        proxy_ic_status="exploratory_heldout_complexity", proxy_ic_n=n,
        proxy_ic_k=scores["information_criterion_k"],
        proxy_ic_domain="heldout_equation_residual" if pde else "heldout_prediction",
        proxy_ic_reason="Exploratory Gaussian score; complexity + variance is not a fitted parameter count; held-out error is not training likelihood.")
    return result


def supplement_cloud_row(row):
    return {**training_ic(row), **proxy_ic(row)}
