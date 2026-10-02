"""Cooperative fit deadlines and bounded adapters; imports models only on demand."""

from __future__ import annotations

import contextlib
import math
import time
from contextvars import ContextVar

_ACTIVE = ContextVar("kd_fit_deadline", default=None)


class SoftDeadline(Exception):
    """Control flow for a cooperative stop, never a fabricated successful fit."""


class FitDeadline:
    def __init__(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Fit deadline must be finite and nonnegative")
        self.expires = time.perf_counter() + seconds
        self.stopped = False

    def remaining(self):
        return max(0.0, self.expires - time.perf_counter())

    def expired(self):
        return self.remaining() == 0.0

    def check(self):
        if self.expired():
            self.stopped = True
            raise SoftDeadline("Fit time limit reached")


def current_deadline():
    return _ACTIVE.get()


@contextlib.contextmanager
def deadline_scope(seconds=None):
    deadline = FitDeadline(seconds) if seconds is not None else None
    token = _ACTIVE.set(deadline)
    try:
        yield deadline
    finally:
        _ACTIVE.reset(token)


def check_deadline():
    deadline = current_deadline()
    if deadline is not None:
        deadline.check()


def _gplearn_fit(function, model, deadline, args, kwargs):
    """Keep upstream evolution and RNG draws identical across warm-start calls."""
    import numpy as np
    from sklearn.utils import check_random_state

    generations, warm_start, seed = model.generations, model.warm_start, model.random_state
    rng = check_random_state(seed)
    initial_state = rng.get_state()
    final_rng = None
    best = None
    reason = "budget_exhausted"
    prior = len(getattr(model, "_programs", [])) if warm_start else 0
    try:
        model.warm_start = True
        if not warm_start:
            for attribute in ("_programs", "run_details_"):
                if hasattr(model, attribute):
                    delattr(model, attribute)
        for generation in range(prior + 1, generations + 1):
            try:
                deadline.check()
            except SoftDeadline:
                reason = "time_limit"
                break
            model.generations = generation
            # gplearn replays earlier seed draws on every warm-start fit.
            # Cloning the initial state also supports RandomState and seed=None.
            final_rng = np.random.RandomState()
            final_rng.set_state(initial_state)
            model.random_state = final_rng
            function(*args, **kwargs)
            candidate = model._program
            fitness = candidate.raw_fitness_
            greater = model._metric.greater_is_better
            if np.isfinite(fitness) and (best is None or
                    (fitness > best.raw_fitness_ if greater else fitness < best.raw_fitness_)):
                best = candidate
            if (fitness >= model.stopping_criteria if greater else
                    fitness <= model.stopping_criteria):
                reason = "early_stop"
                break
            if deadline.expired():
                deadline.stopped = True
                reason = "time_limit"
                break
        if reason == "time_limit":
            if best is None:
                raise RuntimeError("Fit time limit reached before a valid gplearn candidate")
            model._program = best
        model.stop_reason_ = reason
        model.stop_reason_source_ = "generation_deadline" if reason == "time_limit" else "adapter"
        return model
    finally:
        model.generations, model.warm_start, model.random_state = generations, warm_start, seed
        if final_rng is not None and (seed is None or isinstance(seed, np.random.RandomState)):
            rng.set_state(final_rng.get_state())


def _pysr_deadline_condition(model, deadline, julia):
    """Anchor Julia's stop predicate to fit entry, before search initialization."""
    native_now = int(julia.seval("time_ns()"))
    cutoff = native_now + int(deadline.remaining() * 1_000_000_000)
    previous = getattr(model, "early_stop_condition", None)
    expression = "nothing" if previous is None else str(previous)
    return (
        f"let previous = begin {expression} end; cutoff = UInt64({cutoff}); "
        "(loss, complexity) -> (time_ns() >= cutoff || "
        "(previous !== nothing && (previous isa Real ? "
        "loss < previous : previous(loss, complexity)))) end"
    )


def _native_fit(function, model, deadline, parameter, args, kwargs):
    deadline.check()
    original = getattr(model, parameter)
    remaining = deadline.remaining()
    limit = math.ceil(remaining) if parameter == "max_time" else remaining
    if original is not None:
        limit = min(limit, original)
    setattr(model, parameter, limit)
    early_stop = getattr(model, "early_stop_condition", None)
    try:
        if parameter == "timeout_in_seconds":
            from pysr import jl
            model.early_stop_condition = _pysr_deadline_condition(model, deadline, jl)
            deadline.check()
        result = function(*args, **kwargs)
        # The native API exposes no reliable termination code. Report the
        # observed fit deadline, with a distinct source, not guessed convergence.
        if deadline.expired():
            deadline.stopped = True
            model.stop_reason_ = "time_limit"
            model.stop_reason_source_ = "fit_deadline"
        return result
    finally:
        setattr(model, parameter, original)
        if parameter == "timeout_in_seconds":
            model.early_stop_condition = early_stop


class _DeadlineLogger:
    """Use PhySO's public logger hook after it has retained the completed batch."""

    def __init__(self, logger, deadline):
        self.logger = logger
        self.deadline = deadline
        self.stop_reason_ = None
        self.stop_reason_source_ = "epoch_deadline"

    def __getattr__(self, name):
        return getattr(self.logger, name)

    def log(self, *args, **kwargs):
        self.logger.log(*args, **kwargs)
        if self.deadline.expired():
            self.stop_reason_ = "time_limit"
            self.deadline.check()


def _physo_fit(function, deadline, args, kwargs):
    deadline.check()
    options = dict(kwargs)
    factory = options.get("get_run_logger")
    if factory is None:
        from physo.task.args_handler import get_default_run_logger
        factory = get_default_run_logger
    holder = []

    def get_logger():
        logger = factory()
        if logger is None:
            raise ValueError("PhySO soft deadline requires a run logger")
        wrapped = _DeadlineLogger(logger, deadline)
        holder.append(wrapped)
        return wrapped

    options["get_run_logger"] = get_logger
    try:
        return function(*args, **options)
    except SoftDeadline:
        if not holder:
            raise RuntimeError("Fit time limit reached before a PhySO candidate") from None
        logger = holder[0]
        _, programs, rewards, _ = logger.get_pareto_front()
        candidates = [(float(reward), program) for reward, program in zip(rewards, programs)
                      if program is not None and math.isfinite(float(reward)) and reward > 0]
        if not candidates:
            raise RuntimeError("Fit time limit reached before a valid PhySO candidate") from None
        return max(candidates, key=lambda pair: pair[0])[1], logger


def anytime_call(function, *args, **kwargs):
    deadline = current_deadline()
    if deadline is None:
        return function(*args, **kwargs)
    model = getattr(function, "__self__", None)
    module = getattr(type(model), "__module__", "") if model is not None else ""
    if module.startswith("gplearn.") and function.__name__ == "fit":
        return _gplearn_fit(function, model, deadline, args, kwargs)
    if module.startswith("pysr.") and function.__name__ == "fit":
        return _native_fit(function, model, deadline, "timeout_in_seconds", args, kwargs)
    if module.startswith("pyoperon.") and function.__name__ == "fit":
        return _native_fit(function, model, deadline, "max_time", args, kwargs)
    if getattr(function, "__module__", "").startswith("physo.") and function.__name__ == "SR":
        return _physo_fit(function, deadline, args, kwargs)
    return function(*args, **kwargs)
