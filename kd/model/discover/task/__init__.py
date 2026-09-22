"""Lazy exports for DISCOVER task implementations.

Keeping this package initializer lazy prevents ``program -> functions -> task``
from importing ``task.task`` before ``Program`` has finished initializing.
"""

_EXPORTS = {
    "make_task",
    "set_task",
    "Task",
    "HierarchicalTask",
    "SequentialTask",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import task as task_module

    value = getattr(task_module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORTS)


__all__ = sorted(_EXPORTS)
