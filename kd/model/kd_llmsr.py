"""LLM-SR adapter for generic tabular symbolic-regression problems.

The search follows the LLM-SR searcher shipped with LLM-SRBench
(``deep-symbolic-mathematics/llm-srbench``, ``methods/llmsr``): equation
program skeletons are sampled from an OpenAI-compatible chat endpoint, their
free constants ``params[0..MAX_NPARAMS-1]`` are fitted with SciPy BFGS from an
all-ones start in isolated evaluator processes with a per-program timeout, and
scored programs are kept in a multi-island experience buffer with score
clusters.  The adapter exposes the ``fit`` / ``predict`` interface used by this
project's benchmark.

Generated function bodies are never executed directly.  A restricted subset of
Python (assignments, augmented assignments, tuple unpacking and one final
``return``) is inlined into one NumPy expression that is validated before
evaluation.
"""

from __future__ import annotations

import ast
import copy
import keyword
import math
import multiprocessing
import os
import re
import signal
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing.connection import wait

import numpy as np
import requests
from scipy.optimize import minimize
from scipy.special import softmax

DEFAULT_BACKBONE = "meta-llama/Llama-3.1-8B-Instruct"

_FUNCTIONS = {
    "abs": np.abs,
    "arccos": np.arccos,
    "arcsin": np.arcsin,
    "arctan": np.arctan,
    "cos": np.cos,
    "cosh": np.cosh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "log1p": np.log1p,
    "log2": np.log2,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "power": np.power,
    "sign": np.sign,
    "sin": np.sin,
    "sinh": np.sinh,
    "sqrt": np.sqrt,
    "square": np.square,
    "tan": np.tan,
    "tanh": np.tanh,
}
_CONSTANTS = {"pi": np.pi, "e": np.e}
_PARAMETER = re.compile(r"\bp([0-9]+)\b")
_RESERVED_NAMES = {"np", "p", "params", "MAX_NPARAMS", "numpy"}
_MAX_EXPRESSION_LENGTH = 20000
# LLM-SRBench sends this instruction before the program prompt.
_INSTRUCTION = (
    "You are a helpful assistant tasked with discovering mathematical function structures "
    "for scientific systems. Complete the 'equation' function below, considering the "
    "physical meaning and relationships of inputs.\n\n"
)


@dataclass
class _Program:
    body: str
    expression: str
    parameters: np.ndarray
    mse: float
    call_index: int | None  # None for the initial linear skeleton.

    @property
    def score(self):
        return -self.mse


def _softmax(logits, temperature):
    """Tempered softmax whose largest entry absorbs rounding (as upstream)."""
    logits = np.asarray(logits, dtype=float)
    if not np.all(np.isfinite(logits)):
        raise ValueError("Cluster scores must be finite")
    with np.errstate(over="ignore"):
        result = softmax(logits / temperature)
    index = int(np.argmax(result))
    result[index] = 1 - np.sum(result[:index]) - np.sum(result[index + 1 :])
    return result


class _Cluster:
    """Programs on one island that share the same score signature."""

    def __init__(self, program):
        self.score = program.score
        self.programs = [program]

    def sample(self, rng):
        # Upstream prefers shorter programs within a cluster.
        lengths = np.array([len(program.body) for program in self.programs], dtype=float)
        normalized = (lengths - lengths.min()) / (lengths.max() + 1e-6)
        return self.programs[rng.choice(len(self.programs), p=_softmax(-normalized, 1.0))]


class _Island:
    def __init__(self):
        self.clusters = {}
        self.num_programs = 0

    def register(self, program):
        if program.score in self.clusters:
            self.clusters[program.score].programs.append(program)
        else:
            self.clusters[program.score] = _Cluster(program)
        self.num_programs += 1

    def examples(self, rng, functions_per_prompt, temperature_init, temperature_period):
        """Sample in-context programs, ordered from worst to best score."""
        signatures = list(self.clusters)
        temperature = temperature_init * (
            1 - (self.num_programs % temperature_period) / temperature_period
        )
        probabilities = _softmax(signatures, temperature)
        count = min(len(signatures), functions_per_prompt)
        chosen = rng.choice(len(signatures), size=count, p=probabilities)
        programs = [self.clusters[signatures[index]].sample(rng) for index in chosen]
        return sorted(programs, key=lambda program: program.score)


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
        if index.value < 0:
            raise ValueError("Parameter indices must be non-negative")
        return ast.copy_location(ast.Name(id=f"p{index.value}", ctx=ast.Load()), node)


class _SubstituteLocals(ast.NodeTransformer):
    def __init__(self, local_values):
        self.local_values = local_values

    def visit_Name(self, node):  # noqa: N802 - required by ast.NodeTransformer
        if isinstance(node.ctx, ast.Load) and node.id in self.local_values:
            return copy.deepcopy(self.local_values[node.id])
        return node


def _extract_body(sample):
    """Drop text before the first top-level ``def`` and the signature line."""
    lines = sample.splitlines()
    for index, line in enumerate(lines):
        if line[:3] == "def":
            return "\n".join(lines[index + 1 :]) + "\n"
    return sample


def _trim_function_body(generated_code):
    """Keep the function body up to the first syntax error, as upstream does."""
    if not generated_code:
        return ""
    code = f"def fake_function_header():\n{generated_code}"
    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            if exc.lineno is None:
                return ""
            lines = code.splitlines()
            code = "\n".join(lines[: min(exc.lineno - 1, len(lines) - 1)])
    if not tree.body:
        return ""
    return "\n".join(code.splitlines()[1 : tree.body[0].end_lineno]).strip("\n")


def _parameter_slice(node):
    """Return ``params[a:b]`` indices for tuple unpacking, otherwise ``None``."""
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"p", "params"}
        and isinstance(node.slice, ast.Slice)
        and node.slice.step is None
    ):
        return None
    bounds = []
    for bound, default in ((node.slice.lower, 0), (node.slice.upper, None)):
        if bound is None:
            bounds.append(default)
        elif isinstance(bound, ast.Constant) and isinstance(bound.value, int):
            bounds.append(bound.value)
        else:
            raise ValueError("Parameter slices must use integer literals")
    if bounds[1] is None:
        raise ValueError("Parameter slices must have an explicit upper bound")
    return list(range(*bounds))


def _inline_body(body):
    """Inline a restricted function body into its single return expression."""
    tree = ast.parse("def _equation():\n" + body)
    local_values = {}

    def substitute(node):
        result = _SubstituteLocals(local_values).visit(copy.deepcopy(node))
        # Repeated self-substitution (``a = a * a``) grows exponentially.
        if sum(1 for _ in ast.walk(result)) > _MAX_EXPRESSION_LENGTH // 4:
            raise ValueError("Inlined expression is too long")
        return result

    def bind(target, value):
        if isinstance(target, ast.Name):
            local_values[target.id] = value
            return
        if not isinstance(target, (ast.Tuple, ast.List)):
            raise ValueError("Only simple names can be assigned")
        indices = _parameter_slice(value)
        if indices is not None:
            values = [
                ast.Subscript(
                    value=ast.Name(id="params", ctx=ast.Load()),
                    slice=ast.Constant(value=index),
                    ctx=ast.Load(),
                )
                for index in indices
            ]
        elif isinstance(value, (ast.Tuple, ast.List)):
            values = list(value.elts)
        else:
            raise ValueError("Tuple assignment requires a tuple value or a params slice")
        if len(values) != len(target.elts):
            raise ValueError("Tuple assignment has mismatched lengths")
        for element, element_value in zip(target.elts, values):
            bind(element, element_value)

    for statement in tree.body[0].body:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue  # Docstrings and bare constants have no effect.
        if isinstance(statement, ast.Import):
            if all(alias.name == "numpy" and alias.asname in {None, "np"} for alias in statement.names):
                continue
            raise ValueError("Only 'import numpy as np' is allowed")
        if isinstance(statement, ast.Return):
            if statement.value is None:
                raise ValueError("Return statement has no value")
            expression = ast.unparse(substitute(statement.value))
            if len(expression) > _MAX_EXPRESSION_LENGTH:
                raise ValueError("Inlined expression is too long")
            return expression
        if isinstance(statement, ast.Assign):
            value = substitute(statement.value)
            for target in statement.targets:
                bind(target, value)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            bind(statement.target, substitute(statement.value))
        elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
            current = substitute(ast.Name(id=statement.target.id, ctx=ast.Load()))
            local_values[statement.target.id] = ast.BinOp(
                left=current, op=statement.op, right=substitute(statement.value)
            )
        else:
            raise ValueError(f"Unsupported statement: {type(statement).__name__}")
    raise ValueError("Program has no return statement")


def _evaluate_expression(code, X, parameters, variable_names):
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


def _fit_program(expression, parameter_count, X, y, variable_names):
    """Upstream objective: minimize the training MSE with BFGS from all ones."""
    code = compile(expression, "<llmsr-program>", "eval")

    def loss(parameters):
        prediction = _evaluate_expression(code, X, parameters, variable_names)
        return float(np.mean((prediction - y) ** 2))

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        if parameter_count:
            result = minimize(loss, np.ones(parameter_count), method="BFGS")
            parameters, mse = np.asarray(result.x, dtype=float), float(result.fun)
        else:
            parameters = np.empty(0)
            mse = loss(parameters)
    if not math.isfinite(mse):
        raise ValueError("Program loss is not finite")
    return parameters, mse


def _evaluation_worker(connection, expression, parameter_count, X, y, variable_names, timeout):
    if hasattr(signal, "SIGALRM"):
        # Backstop so evaluators cannot outlive a killed benchmark worker.
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(int(math.ceil(timeout)) + 10)
    try:
        parameters, mse = _fit_program(expression, parameter_count, X, y, variable_names)
        connection.send(("ok", parameters, mse))
    except Exception as exc:  # Every failure only rejects this program.
        connection.send(("error", f"{type(exc).__name__}: {exc}"[:200], None))
    finally:
        connection.close()


class KD_LLMSR:
    """Discover an expression by evolving LLM-generated program skeletons.

    The defaults reproduce the LLM-SRBench LLM-SR protocol except for the
    backbone endpoint, which must be configured.  ``max_llm_calls`` counts
    completed chat requests; each request returns one program, so the default
    budget is 250 prompts with ``samples_per_prompt=4`` programs each.

    API keys are never accepted as constructor arguments.  ``api_key_env``
    names the environment variable read immediately before an API request, so
    secrets do not enter benchmark manifests or result files.
    """

    def __init__(
        self,
        endpoint=None,
        use_api=True,
        api_model=None,
        api_key_env="LLMSR_API_KEY",
        max_llm_calls=1000,
        samples_per_prompt=4,
        num_islands=10,
        functions_per_prompt=2,
        num_evaluators=4,
        evaluation_timeout=30,
        max_parameters=10,
        prompt_samples=0,
        temperature=0.8,
        top_p=None,
        max_tokens=1024,
        request_timeout=120,
        max_request_retries=3,
        request_seed=True,
        reset_period=4 * 60 * 60,
        cluster_sampling_temperature_init=0.1,
        cluster_sampling_temperature_period=30000,
        random_state=None,
    ):
        self.endpoint = endpoint
        self.use_api = use_api
        self.api_model = api_model
        self.api_key_env = api_key_env
        self.max_llm_calls = max_llm_calls
        self.samples_per_prompt = samples_per_prompt
        self.num_islands = num_islands
        self.functions_per_prompt = functions_per_prompt
        self.num_evaluators = num_evaluators
        self.evaluation_timeout = evaluation_timeout
        self.max_parameters = max_parameters
        self.prompt_samples = prompt_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.max_request_retries = max_request_retries
        self.request_seed = request_seed
        self.reset_period = reset_period
        self.cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self.cluster_sampling_temperature_period = cluster_sampling_temperature_period
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
            if (
                name in used
                or name in _FUNCTIONS
                or name in _RESERVED_NAMES
                or keyword.iskeyword(name)
                or _PARAMETER.fullmatch(name)
                or name.startswith("equation")
            ):
                name = f"x{index + 1}"
            used.add(name)
            names.append(name)
        return names

    def _validate_options(self):
        positive = {
            "max_llm_calls": self.max_llm_calls,
            "samples_per_prompt": self.samples_per_prompt,
            "num_islands": self.num_islands,
            "functions_per_prompt": self.functions_per_prompt,
            "num_evaluators": self.num_evaluators,
            "max_parameters": self.max_parameters,
            "max_tokens": self.max_tokens,
            "cluster_sampling_temperature_period": self.cluster_sampling_temperature_period,
        }
        non_negative = {
            "prompt_samples": self.prompt_samples,
            "max_request_retries": self.max_request_retries,
        }
        for name, value in {**positive, **non_negative}.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < (1 if name in positive else 0):
                raise ValueError(f"{name} must be {'positive' if name in positive else 'non-negative'}")
        for name in (
            "evaluation_timeout",
            "request_timeout",
            "reset_period",
            "cluster_sampling_temperature_init",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if not self.temperature >= 0:
            raise ValueError("temperature must be non-negative")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

    def _connection(self):
        # No implicit remote default: prompts must only go to a configured service.
        endpoint = self.endpoint or os.environ.get("LLMSR_ENDPOINT")
        if not endpoint:
            raise RuntimeError(
                "LLM-SR requires a completion endpoint; set LLMSR_ENDPOINT "
                "or pass endpoint in the baseline configuration"
            )
        if self.use_api:
            return endpoint, self.api_model or os.environ.get("LLMSR_MODEL") or DEFAULT_BACKBONE
        return endpoint, None

    def _build_preface(self, X, y, target_name):
        """Build the problem header that precedes the versioned equation programs."""
        output = str(target_name) if target_name else "y"
        names = self.variable_names_
        inputs = names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", and " + names[-1]
        data = ""
        if self.prompt_samples:
            # Sort before taking evenly spaced rows so the prompt covers the
            # observed response range without sending the complete training set.
            order = np.argsort(y, kind="stable")
            count = min(self.prompt_samples, len(order))
            rows = []
            for row in order[np.linspace(0, len(order) - 1, count, dtype=int)]:
                values = ", ".join(f"{name}={X[row, column]:.8g}" for column, name in enumerate(names))
                rows.append(f"{values} -> {output}={y[row]:.8g}")
            data = "Representative training observations:\n" + "\n".join(rows) + "\n"
        self._preface_ = (
            '"""\n'
            f"Find the mathematical function skeleton that represents {output}, "
            f"given data on {inputs}.\n{data}"
            '"""\n\n\n'
            "import numpy as np\n\n"
            "#Initialize parameters\n"
            f"MAX_NPARAMS = {self.max_parameters}\n"
            "params = [1.0]*MAX_NPARAMS\n\n\n"
        )
        arguments = "\n".join(
            f"        {name}: A numpy array representing observations of {name}." for name in names
        )
        self._first_docstring_ = (
            f" Mathematical function for {output}\n\n"
            "    Args:\n"
            f"{arguments}\n"
            "        params: Array of numeric constants or parameters to be optimized\n\n"
            "    Return:\n"
            f"        A numpy array representing {output} as the result of applying the "
            "mathematical function to the inputs.\n    "
        )
        self._signature_ = ", ".join(f"{name}: np.ndarray" for name in names)
        self._signature_ += ", params: np.ndarray"

    def _build_prompt(self, examples):
        functions = []
        for index, program in enumerate(examples):
            docstring = (
                self._first_docstring_
                if index == 0
                else f"Improved version of `equation_v{index - 1}`."
            )
            functions.append(
                f"def equation_v{index}({self._signature_}) -> np.ndarray:\n"
                f'    """{docstring}"""\n{program.body}\n\n'
            )
        index = len(examples)
        functions.append(
            f"def equation_v{index}({self._signature_}) -> np.ndarray:\n"
            f'    """Improved version of `equation_v{index - 1}`."""\n\n'
        )
        return _INSTRUCTION + "\n" + self._preface_ + "\n" + "\n".join(functions)

    def _request_payload(self, prompt, call_index):
        if not self.use_api:
            # This shape matches the completion server shipped by upstream LLM-SR.
            params = {
                "do_sample": True,
                "temperature": self.temperature,
                "max_new_tokens": self.max_tokens,
                "add_special_tokens": False,
                "skip_special_tokens": True,
            }
            if self.top_p is not None:
                params["top_p"] = self.top_p
            return {"prompt": prompt, "repeat_prompt": 1, "params": params}
        payload = {
            "model": self._model_,
            "messages": [{"role": "user", "content": prompt}],
            "n": 1,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.request_seed and self.random_state is not None:
            payload["seed"] = (int(self.random_state) * 1_000_003 + call_index) % 2**31
        return payload

    @staticmethod
    def _completion_text(data):
        if "choices" in data:
            choices = data["choices"]
            if not choices:
                raise ValueError("Endpoint returned no choices")
            choice = choices[0]
            text = (choice.get("message") or {}).get("content", choice.get("text"))
        else:
            text = data.get("content", data.get("generated_text", data.get("response")))
            if isinstance(text, list):
                text = text[0] if text else None
        if text is None:
            return ""
        if not isinstance(text, str):
            raise ValueError("Endpoint returned a non-text completion")
        return text

    def _request_completion(self, prompt, call_index):
        """Return one completion; retry transient failures a bounded number of times."""
        headers = {"Content-Type": "application/json"}
        if self.use_api:
            key = next(
                (
                    os.environ[name]
                    for name in (self.api_key_env, "VLLM_API_KEY", "OPENAI_API_KEY")
                    if os.environ.get(name)
                ),
                None,
            )
            if key:
                headers["Authorization"] = f"Bearer {key}"
        payload = self._request_payload(prompt, call_index)
        for attempt in range(self.max_request_retries + 1):
            try:
                response = requests.post(
                    self._endpoint_,
                    json=payload,
                    headers=headers,
                    timeout=float(self.request_timeout),
                )
                response.raise_for_status()
                data = response.json()
                return self._completion_text(data), data, attempt
            except (requests.RequestException, ValueError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                permanent = status is not None and 400 <= status < 500 and status != 429
                if permanent or attempt == self.max_request_retries:
                    raise RuntimeError(
                        f"LLM-SR request {call_index} to {self._endpoint_!r} failed after "
                        f"{attempt + 1} attempt(s): {exc}"
                    ) from exc
                time.sleep(min(2.0**attempt, 30.0))

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
                    and (node.attr in _FUNCTIONS or node.attr in _CONSTANTS)
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
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed")
        expression = ast.unparse(tree.body)
        indices = [int(value) for value in _PARAMETER.findall(expression)]
        parameter_count = max(indices, default=-1) + 1
        if parameter_count > self.max_parameters:
            raise ValueError("Candidate uses more free constants than MAX_NPARAMS")
        return expression, compile(tree, "<llmsr-expression>", "eval"), parameter_count

    def _evaluate_parallel(self, jobs, X, y):
        """Fit programs in worker processes, ``num_evaluators`` at a time.

        Each job is ``(expression, parameter_count)``.  Returns one
        ``(status, parameters, mse_or_message)`` tuple per job, where status is
        ``ok``, ``error`` or ``timeout``.
        """
        context = multiprocessing.get_context()
        results = [None] * len(jobs)
        pending = list(enumerate(jobs))
        running = {}
        while pending or running:
            while pending and len(running) < self.num_evaluators:
                index, (expression, count) = pending.pop(0)
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(
                    target=_evaluation_worker,
                    args=(
                        sender,
                        expression,
                        count,
                        X,
                        y,
                        self.variable_names_,
                        self.evaluation_timeout,
                    ),
                    daemon=True,
                )
                process.start()
                sender.close()
                running[receiver] = (index, process, time.monotonic() + self.evaluation_timeout)
            next_deadline = min(deadline for _, _, deadline in running.values())
            for receiver in wait(list(running), timeout=max(0.0, next_deadline - time.monotonic())):
                index, process, _ = running.pop(receiver)
                try:
                    status, parameters, value = receiver.recv()
                    results[index] = (status, parameters, value)
                except EOFError:
                    results[index] = ("error", None, "Evaluator exited without a result")
                receiver.close()
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join()
            now = time.monotonic()
            for receiver, (index, process, deadline) in list(running.items()):
                if now >= deadline:
                    running.pop(receiver)
                    process.terminate()
                    process.join()
                    receiver.close()
                    results[index] = ("timeout", None, "Evaluation timed out")
        return results

    def _evaluate_samples(self, samples, X, y, stats):
        """Parse and fit ``(body, call_index)`` samples; return accepted programs."""
        jobs, parsed = [], []
        for body, call_index in samples:
            try:
                expression = _inline_body(body)
                expression, _, count = self._compile_expression(expression, self.variable_names_)
            except (SyntaxError, TypeError, ValueError, RecursionError) as exc:
                stats["rejected_parse"] += 1
                self._rejections[f"parse: {type(exc).__name__}: {exc}"[:120]] += 1
                continue
            jobs.append((expression, count))
            parsed.append((body, expression, call_index))
        programs = []
        for (body, expression, call_index), (status, parameters, value) in zip(
            parsed, self._evaluate_parallel(jobs, X, y)
        ):
            if status == "ok":
                programs.append(_Program(body, expression, parameters, value, call_index))
            elif status == "timeout":
                stats["evaluation_timeouts"] += 1
            else:
                stats["rejected_evaluation"] += 1
                self._rejections[f"evaluation: {value}"[:120]] += 1
        stats["accepted"] += len(programs) - sum(p.call_index is None for p in programs)
        return programs

    def _register(self, program, island_id, stats):
        self._islands[island_id].register(program)
        best = self._island_best[island_id]
        if best is None or program.score > best.score:
            self._island_best[island_id] = program
        if self._best is None or program.mse < self._best.mse:
            self._best = program
        if time.monotonic() - self._last_reset > self.reset_period:
            self._last_reset = time.monotonic()
            self._reset_islands()
            stats["island_resets"] += 1

    def _reset_islands(self):
        """Reset the weaker half of islands from the best programs of the others."""
        scores = np.array([program.score for program in self._island_best])
        order = np.argsort(scores + self._rng.normal(size=len(scores)) * 1e-6)
        reset = order[: self.num_islands // 2]
        keep = order[self.num_islands // 2 :]
        for island_id in reset:
            self._islands[island_id] = _Island()
            self._island_best[island_id] = None
            founder = self._island_best[int(self._rng.choice(keep))]
            self._islands[island_id].register(founder)
            self._island_best[island_id] = founder

    @staticmethod
    def _format_expression(program):
        values = {f"p{i}": f"({value:.12g})" for i, value in enumerate(program.parameters)}
        expression = _PARAMETER.sub(lambda match: values[match.group(0)], program.expression)
        return expression.replace("np.", "")

    def fit(self, X, y, variable_names=None, target_name=None):
        """Run LLM-guided structure search and fit free constants on ``X, y``."""
        self._validate_options()
        # Fail before numerical work if the service is unconfigured.
        self._endpoint_, self._model_ = self._connection()
        X = np.asarray(X, dtype=float)
        y = np.ravel(np.asarray(y, dtype=float))
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or y.shape != (X.shape[0],):
            raise ValueError("X must be 2D and y must contain one value per row")
        if X.shape[0] < 2 or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("LLM-SR requires at least two finite samples")
        self.variable_names_ = self._variable_names(X.shape[1], variable_names)
        self._rng = np.random.default_rng(self.random_state)
        self._rejections = Counter()
        self._build_preface(X, y, target_name)
        stats = Counter()

        # Upstream starts from a linear skeleton; keep it valid when a dataset
        # has more inputs than the free-constant budget.
        seed_names = self.variable_names_[: self.max_parameters - 1]
        terms = [f"params[{index}] * {name}" for index, name in enumerate(seed_names)]
        initial_body = (
            f"    output = {' + '.join(terms + [f'params[{len(seed_names)}]'])}\n"
            "    return output"
        )
        initial = self._evaluate_samples([(initial_body, None)], X, y, stats)
        if not initial:
            raise RuntimeError(f"LLM-SR initial skeleton failed: {dict(self._rejections)}")
        self._islands = [_Island() for _ in range(self.num_islands)]
        self._island_best = [None] * self.num_islands
        self._best = None
        self._last_reset = time.monotonic()
        for island_id in range(self.num_islands):
            self._register(initial[0], island_id, stats)

        calls = 0
        served_models = set()
        stopped_on_zero_loss = False
        with ThreadPoolExecutor(max_workers=self.samples_per_prompt) as executor:
            while calls < self.max_llm_calls:
                if self._best.mse == 0:
                    stopped_on_zero_loss = True  # Upstream stops on a perfect score.
                    break
                island_id = int(self._rng.integers(self.num_islands))
                examples = self._islands[island_id].examples(
                    self._rng,
                    self.functions_per_prompt,
                    self.cluster_sampling_temperature_init,
                    self.cluster_sampling_temperature_period,
                )
                prompt = self._build_prompt(examples)
                count = min(self.samples_per_prompt, self.max_llm_calls - calls)
                indices = range(calls, calls + count)
                responses = list(executor.map(partial(self._request_completion, prompt), indices))
                calls += count
                stats["prompts"] += 1
                samples = []
                for call_index, (text, data, retries) in zip(indices, responses):
                    stats["request_retries"] += retries
                    usage = data.get("usage") or {}
                    stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                    stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                    if data.get("model"):
                        served_models.add(str(data["model"]))
                    samples.append((_trim_function_body(_extract_body(text)), call_index))
                for program in self._evaluate_samples(samples, X, y, stats):
                    self._register(program, island_id, stats)

        best = self._best
        self._expression_code_ = self._compile_expression(best.expression, self.variable_names_)[1]
        self.parameters_ = best.parameters.copy()
        self.best_expression_ = self._format_expression(best)
        self.best_program_ = (
            f"def equation({self._signature_}) -> np.ndarray:\n{best.body}\n"
        )
        self.training_mse_ = best.mse
        self.stop_reason_ = "early_stop" if stopped_on_zero_loss else "budget_exhausted"
        self.search_stats_ = {
            "method": "LLM-SR (LLM-SRBench searcher protocol)",
            "backbone": self._model_,
            "served_models": sorted(served_models),
            "llm_calls": calls,
            "llm_call_budget": self.max_llm_calls,
            "prompts": stats["prompts"],
            "request_retries": stats["request_retries"],
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"],
            "accepted": stats["accepted"],
            "rejected_parse": stats["rejected_parse"],
            "rejected_evaluation": stats["rejected_evaluation"],
            "evaluation_timeouts": stats["evaluation_timeouts"],
            "island_resets": stats["island_resets"],
            "islands": self.num_islands,
            "stopped_on_zero_loss": stopped_on_zero_loss,
            "best_call_index": best.call_index,
            "training_mse": best.mse,
            "rejection_reasons": dict(self._rejections.most_common(20)),
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
        return _evaluate_expression(self._expression_code_, X, self.parameters_, self.variable_names_)
