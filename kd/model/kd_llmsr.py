"""LLM-SR adapter for generic tabular symbolic-regression problems.

The upstream LLM-SR implementation couples its search pipeline to four bundled
datasets.  This adapter keeps the same program-skeleton idea, island-based
candidate search, and numerical constant fitting while exposing the ``fit`` /
``predict`` interface used by this project's benchmark.
"""

from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass

import numpy as np
import requests
from scipy.optimize import least_squares

_FUNCTIONS = {
    "abs": np.abs,
    "cos": np.cos,
    "exp": np.exp,
    "log": np.log,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "square": np.square,
    "tan": np.tan,
    "tanh": np.tanh,
}
_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_PARAMETER = re.compile(r"\bp([0-9]+)\b")


@dataclass
class _Candidate:
    expression: str
    parameters: np.ndarray
    mse: float


class _ParameterSubscriptToName(ast.NodeTransformer):
    """Convert the upstream ``params[i]`` convention to the local ``pi`` form."""

    def visit_Subscript(self, node):  # noqa: N802 - required by ast.NodeTransformer
        node = self.generic_visit(node)
        if not isinstance(node.value, ast.Name) or node.value.id not in {"p", "params"}:
            return node
        index = node.slice
        if hasattr(ast, "Index") and isinstance(index, ast.Index):
            index = index.value
        if not isinstance(index, ast.Constant) or not isinstance(index.value, int):
            raise ValueError("Parameter indices must be integer literals")
        return ast.copy_location(ast.Name(id=f"p{index.value}", ctx=ast.Load()), node)


class KD_LLMSR:
    """Discover an expression by evolving LLM-generated program skeletons.

    API keys are never accepted as constructor arguments.  ``api_key_env``
    names the environment variable read immediately before an API request, so
    secrets do not enter benchmark manifests or result files.
    """

    def __init__(
        self,
        endpoint=None,
        use_api=False,
        api_model=None,
        api_key_env="LLMSR_API_KEY",
        max_samples=8,
        samples_per_prompt=2,
        num_islands=2,
        functions_per_prompt=2,
        prompt_samples=12,
        max_parameters=8,
        optimizer_iterations=200,
        optimizer_restarts=3,
        temperature=0.7,
        max_tokens=512,
        request_timeout=120,
        random_state=None,
    ):
        self.endpoint = endpoint
        self.use_api = use_api
        self.api_model = api_model
        self.api_key_env = api_key_env
        self.max_samples = max_samples
        self.samples_per_prompt = samples_per_prompt
        self.num_islands = num_islands
        self.functions_per_prompt = functions_per_prompt
        self.prompt_samples = prompt_samples
        self.max_parameters = max_parameters
        self.optimizer_iterations = optimizer_iterations
        self.optimizer_restarts = optimizer_restarts
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.random_state = random_state

    @staticmethod
    def _variable_names(count, requested):
        if requested is None:
            requested = [f"x{i + 1}" for i in range(count)]
        if len(requested) != count:
            raise ValueError(f"Expected {count} variable names, received {len(requested)}")
        names = []
        used = set()
        for index, value in enumerate(requested):
            name = re.sub(r"\W+", "_", str(value)).strip("_") or f"x{index + 1}"
            if name[0].isdigit():
                name = "x_" + name
            if name in used or name in _FUNCTIONS or name in {"np", "p", "params"}:
                name = f"x{index + 1}"
            used.add(name)
            names.append(name)
        return names

    def _validate_options(self):
        integer_options = {
            "max_samples": self.max_samples,
            "samples_per_prompt": self.samples_per_prompt,
            "num_islands": self.num_islands,
            "functions_per_prompt": self.functions_per_prompt,
            "prompt_samples": self.prompt_samples,
            "max_parameters": self.max_parameters,
            "optimizer_iterations": self.optimizer_iterations,
            "optimizer_restarts": self.optimizer_restarts,
            "max_tokens": self.max_tokens,
        }
        for name, value in integer_options.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")

    def _connection(self):
        endpoint = self.endpoint or os.environ.get("LLMSR_ENDPOINT")
        if self.use_api:
            endpoint = endpoint or "https://api.openai.com/v1/chat/completions"
            model = self.api_model or os.environ.get("LLMSR_MODEL")
            if not model:
                raise RuntimeError(
                    "LLM-SR API mode requires api_model or the LLMSR_MODEL environment variable"
                )
            return endpoint, model
        if not endpoint:
            raise RuntimeError(
                "LLM-SR requires a local completion endpoint; set LLMSR_ENDPOINT "
                "or pass endpoint in the baseline configuration"
            )
        return endpoint, None

    def _build_prompt(self, island, variable_names):
        ranked = sorted(island, key=lambda item: item.mse)
        examples = ranked[: self.functions_per_prompt]
        arguments = ", ".join(variable_names + [f"p{i}" for i in range(self.max_parameters)])
        versions = []
        for index, candidate in enumerate(reversed(examples)):
            versions.append(
                f"def equation_v{index}({arguments}):\n"
                f"    return {candidate.expression}\n"
                f"# training_mse={candidate.mse:.8g}"
            )
        next_index = len(versions)
        history = "\n\n".join(versions)
        return (
            "Discover a concise mathematical equation that predicts y from the supplied "
            f"variables {variable_names}. Representative training observations are:\n"
            f"{self._data_summary_}\n\nImprove the program skeletons below. You may use "
            "arithmetic, sin, cos, tan, tanh, exp, log, sqrt, abs, square, minimum, maximum, "
            f"and free constants p0 through p{self.max_parameters - 1}. Return exactly one "
            "Python function and put the candidate expression directly in its return statement. "
            "Do not import modules or access data outside the function.\n\n"
            f"{history}\n\n"
            f"def equation_v{next_index}({arguments}):\n"
            "    # Return an improved expression here."
        )

    def _request_samples(self, prompt, count):
        endpoint, model = self._connection()
        headers = {"Content-Type": "application/json"}
        if self.use_api:
            key = os.environ.get(self.api_key_env) or os.environ.get("OPENAI_API_KEY")
            if key:
                headers["Authorization"] = f"Bearer {key}"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "n": count,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        else:
            # This shape matches the completion server shipped by upstream LLM-SR.
            payload = {
                "prompt": prompt,
                "repeat_prompt": count,
                "params": {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "add_special_tokens": False,
                    "skip_special_tokens": True,
                },
            }
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=float(self.request_timeout),
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"LLM-SR request to {endpoint!r} failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"LLM-SR endpoint {endpoint!r} returned invalid JSON") from exc

        if "choices" in data:
            samples = [
                choice.get("message", {}).get("content", choice.get("text", ""))
                for choice in data["choices"]
            ]
        else:
            content = data.get("content", data.get("generated_text", data.get("response", [])))
            samples = content if isinstance(content, list) else [content]
        samples = [sample for sample in samples if isinstance(sample, str) and sample.strip()]
        if not samples:
            raise RuntimeError("LLM-SR endpoint returned no text completions")
        return samples

    @staticmethod
    def _extract_expression(sample):
        fences = _CODE_FENCE.findall(sample)
        text = fences[0] if fences else sample.strip()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            match = re.search(r"\breturn\s+([^\n#]+)", text)
            expression = match.group(1).strip() if match else text.splitlines()[0].strip()
            return expression.replace("^", "**")
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if functions:
            returns = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Return)]
            if len(returns) != 1 or returns[0].value is None:
                raise ValueError("Candidate functions must contain one return expression")
            return ast.unparse(returns[0].value)
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
            return ast.unparse(tree.body[0].value)
        raise ValueError("Completion does not contain an equation expression")

    def _compile_expression(self, expression, variable_names):
        tree = ast.parse(expression, mode="eval")
        tree = _ParameterSubscriptToName().visit(tree)
        ast.fix_missing_locations(tree)
        allowed_names = set(variable_names) | set(_FUNCTIONS) | {"np"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if not (_PARAMETER.fullmatch(node.id) or node.id in allowed_names):
                    raise ValueError(f"Unknown name in candidate: {node.id}")
            elif isinstance(node, ast.Attribute):
                if not (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "np"
                    and node.attr in _FUNCTIONS
                ):
                    raise ValueError("Only approved numpy mathematical functions are allowed")
            elif isinstance(node, ast.Call):
                function = node.func
                valid = isinstance(function, ast.Name) and function.id in _FUNCTIONS
                valid = valid or (
                    isinstance(function, ast.Attribute)
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "np"
                    and function.attr in _FUNCTIONS
                )
                if not valid or node.keywords:
                    raise ValueError("Candidate contains an unsupported function call")
            elif not isinstance(
                node,
                (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Constant,
                    ast.Load,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Pow,
                    ast.Mod,
                    ast.USub,
                    ast.UAdd,
                    ast.Name,
                    ast.Attribute,
                    ast.Call,
                ),
            ):
                raise ValueError(f"Unsupported syntax in candidate: {type(node).__name__}")
        expression = ast.unparse(tree.body)
        indices = [int(value) for value in _PARAMETER.findall(expression)]
        parameter_count = max(indices, default=-1) + 1
        if parameter_count > self.max_parameters:
            raise ValueError("Candidate uses more free constants than max_parameters")
        return expression, compile(tree, "<llmsr-expression>", "eval"), parameter_count

    @staticmethod
    def _evaluate(code, X, parameters, variable_names):
        namespace = dict(_FUNCTIONS)
        namespace["np"] = np
        namespace.update({name: X[:, index] for index, name in enumerate(variable_names)})
        namespace.update({f"p{index}": value for index, value in enumerate(parameters)})
        with np.errstate(all="ignore"):
            values = np.asarray(eval(code, {"__builtins__": {}}, namespace), dtype=float)
        if values.ndim == 0:
            values = np.full(X.shape[0], values.item(), dtype=float)
        values = np.ravel(values)
        if values.shape != (X.shape[0],):
            raise ValueError(f"Candidate returned shape {values.shape}, expected {(X.shape[0],)}")
        return values

    def _fit_candidate(self, expression, X, y, variable_names, rng):
        expression, code, count = self._compile_expression(expression, variable_names)

        def residual(parameters):
            try:
                values = self._evaluate(code, X, parameters, variable_names)
                return np.nan_to_num(values - y, nan=1e12, posinf=1e12, neginf=-1e12)
            except (FloatingPointError, OverflowError, ValueError):
                return np.full(y.shape, 1e12)

        starts = [np.ones(count)]
        starts.extend(rng.normal(size=count) for _ in range(self.optimizer_restarts - 1))
        if count:
            fitted = [
                least_squares(residual, start, max_nfev=self.optimizer_iterations)
                for start in starts
            ]
            result = min(fitted, key=lambda item: float(np.mean(residual(item.x) ** 2)))
            parameters = result.x
        else:
            parameters = np.empty(0)
        error = residual(parameters)
        mse = float(np.mean(error**2))
        if not math.isfinite(mse) or mse >= 1e23:
            raise ValueError("Candidate did not produce finite predictions")
        return _Candidate(expression, parameters, mse)

    @staticmethod
    def _format_expression(candidate):
        values = {f"p{i}": f"({value:.12g})" for i, value in enumerate(candidate.parameters)}
        expression = _PARAMETER.sub(lambda match: values[match.group(0)], candidate.expression)
        return expression.replace("np.", "")

    def fit(self, X, y, variable_names=None):
        """Run LLM-guided structure search and fit free constants on ``X, y``."""
        self._validate_options()
        self._connection()  # Fail before numerical work if the service is unconfigured.
        X = np.asarray(X, dtype=float)
        y = np.ravel(np.asarray(y, dtype=float))
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or y.shape != (X.shape[0],):
            raise ValueError("X must be 2D and y must contain one value per row")
        if X.shape[0] < 2 or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("LLM-SR requires at least two finite samples")
        self.variable_names_ = self._variable_names(X.shape[1], variable_names)
        rng = np.random.default_rng(self.random_state)
        # Sort before taking evenly spaced rows so the prompt covers the
        # observed response range without sending the complete training set.
        order = np.argsort(y, kind="stable")
        indices = np.linspace(0, len(order) - 1, min(self.prompt_samples, len(order)), dtype=int)
        prompt_rows = []
        for row_index in order[indices]:
            inputs = ", ".join(
                f"{name}={X[row_index, column]:.8g}"
                for column, name in enumerate(self.variable_names_)
            )
            prompt_rows.append(f"{inputs} -> y={y[row_index]:.8g}")
        self._data_summary_ = "\n".join(prompt_rows)

        # Keep the seed valid even when a dataset has more columns than the
        # configured free-constant budget; later LLM candidates may use any input.
        seed_names = self.variable_names_[: max(0, self.max_parameters - 1)]
        linear = "p0" + "".join(
            f" + p{index + 1} * {name}" for index, name in enumerate(seed_names)
        )
        initial = self._fit_candidate(linear, X, y, self.variable_names_, rng)
        islands = [[initial] for _ in range(self.num_islands)]
        accepted = 0
        rejected = 0
        sampled = 0
        request_index = 0
        while sampled < self.max_samples:
            island_id = request_index % self.num_islands
            request_index += 1
            count = min(self.samples_per_prompt, self.max_samples - sampled)
            prompt = self._build_prompt(islands[island_id], self.variable_names_)
            samples = self._request_samples(prompt, count)
            for sample in samples[:count]:
                sampled += 1
                try:
                    expression = self._extract_expression(sample)
                    candidate = self._fit_candidate(expression, X, y, self.variable_names_, rng)
                except (SyntaxError, TypeError, ValueError):
                    rejected += 1
                    continue
                islands[island_id].append(candidate)
                islands[island_id].sort(key=lambda item: item.mse)
                del islands[island_id][max(2, self.functions_per_prompt * 2) :]
                accepted += 1
            # Some compatible endpoints return fewer completions than requested.
            if len(samples) < count:
                sampled += count - len(samples)

        best = min((candidate for island in islands for candidate in island), key=lambda c: c.mse)
        self._expression_code_ = self._compile_expression(best.expression, self.variable_names_)[1]
        self.parameters_ = best.parameters.copy()
        self.best_expression_ = self._format_expression(best)
        self.training_mse_ = best.mse
        self.search_stats_ = {
            "sampled": sampled,
            "accepted": accepted,
            "rejected": rejected,
            "islands": self.num_islands,
        }
        return self

    def predict(self, X):
        """Evaluate the best fitted program skeleton."""
        if not hasattr(self, "_expression_code_"):
            raise RuntimeError("KD_LLMSR must be fitted before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[1] != len(self.variable_names_):
            raise ValueError(f"Expected X with {len(self.variable_names_)} columns")
        return self._evaluate(self._expression_code_, X, self.parameters_, self.variable_names_)
