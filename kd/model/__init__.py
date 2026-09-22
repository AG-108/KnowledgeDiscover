"""Model implementations, loaded on demand to isolate optional dependencies."""

from importlib import import_module

_MODEL_MODULES = {
    "DLGA": ".dlga",
    "KD_DLGA": ".kd_dlga",
    "KD_DSCV": ".kd_dscv",
    "KD_DSCV_SPR": ".kd_dscv",
    "KD_DeepMoD": ".kd_deepmod",
    "KD_SGA": ".kd_sga",
    "KD_EqGPT": ".kd_eqgpt",
    "KD_SymbolicGPT": ".kd_symbolicgpt",
    "KD_DSO": ".kd_dso",
    "SINDyModel": ".kd_sindy",
    "PySINDyModel": ".kd_sindy",
    "KD_LLMSR": ".kd_llmsr",
    "IntegralWeakPDEModel": ".kd_wsindy",
    "E2ETransformerModel": ".kd_e2e",
}


def __getattr__(name):
    if name not in _MODEL_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODEL_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "DLGA",
    "KD_DLGA",
    "KD_DSCV",
    "KD_DSCV_SPR",
    "KD_DeepMoD",
    "KD_SGA",
    "KD_EqGPT",
    "KD_SymbolicGPT",
    "KD_DSO",
    "SINDyModel",
    "PySINDyModel",
    "KD_LLMSR",
    "IntegralWeakPDEModel",
    "E2ETransformerModel",
]
