"""Fixed upstream identities checked byte-for-byte on 2026-10-02.

The function-set table was additionally verified at the same DSO revision on
2026-10-11 when it became an explicit runtime dependency.

This establishes file lineage, not new physical-unit or redistribution claims.
PIC and DeePyMoD evolution assets were additionally verified on 2026-10-16.
"""

from copy import deepcopy

DISCOVER = "https://github.com/menggedu/DISCOVER"
DISCOVER_REVISION = "6bf37ce110ccbcb07fb14108cc46b185cdf8fc84"
DSO = "https://github.com/dso-org/deep-symbolic-optimization"
DSO_REVISION = "8348d5b08d1eef6170fdbfd222a492ded990ea12"
ASSETS = {
    "data/Parametric_convection_diffusion.npy": (
        "https://github.com/woshixuhao/PIC_code", "0230007c3457022b32b7f38b421ec0f9c817cd5d",
        "data/Parametric_convection_diffusion/variable-GWFE.npy",
        "6d000a7c22e5551e7ab7523aabbcd497d8fd4756d6ce32b53ed692790e164af9"),
    "data/Advection_diffusion.mat": (
        "https://github.com/PhIMaL/DeePyMoD", "543c2e5fc7a6a5637be660775873724d884790ba",
        "examples/data/advection_diffusion.mat",
        "faf14f3b50500c32f08a910e47a5c0c7d017d641818db4d5bb268dd4153ea6d7"),
    "data/benchmarks.csv": (DSO, DSO_REVISION, "dso/dso/task/regression/benchmarks.csv",
                            "42525c6795f3ebc7a6f38d59dbbb011ce92e0ef7d0b533346be8fc0e29abad9a"),
    "data/function_sets.csv": (DSO, DSO_REVISION, "dso/dso/task/regression/function_sets.csv",
                               "559b608b64135e51865b7eaa01d51673dd060e42627163f58a68b8fc058e9bbd"),
    "data/burgers2.mat": (DISCOVER, DISCOVER_REVISION, "dso/dso/task/pde/data_new/burgers2.mat",
                          "a3e94965062ff4fc4b473c0909e0b1fb891dbc740dd7359743f38b351a281b8e"),
    "data/KdV_equation.mat": (DISCOVER, DISCOVER_REVISION, "dso/dso/task/pde/data_new/Kdv.mat",
                             "7c04e89a0c65692105dbe7fc8ba52cb90ce01a64e47e995a5736fbf1c06be220"),
    "data/chafee_infante_CI.npy": (DISCOVER, DISCOVER_REVISION, "dso/dso/task/pde/data_new/chafee_infante_CI.npy",
                                  "9ab264bf39895a73f581813c3687619f61ee01fe5e514627bde885445b6640a0"),
    "data/chafee_infante_x.npy": (DISCOVER, DISCOVER_REVISION, "dso/dso/task/pde/data_new/chafee_infante_x.npy",
                                 "91b28bb9edbecd86abedff439f9e9f0cd4d2d2e239ef68722aa78abcdb9b538a"),
    "data/chafee_infante_t.npy": (DISCOVER, DISCOVER_REVISION, "dso/dso/task/pde/data_new/chafee_infante_t.npy",
                                 "10b0e2b5a647ce58128be9182e5d8943bda5e45ff7e8fd6bb33cd77eefb42f7a"),
}


def evidence(paths):
    selected = {path: ASSETS[path] for path in paths if path in ASSETS}
    dates = {"data/function_sets.csv": "2026-10-11",
             "data/Parametric_convection_diffusion.npy": "2026-10-16",
             "data/Advection_diffusion.mat": "2026-10-16"}
    return deepcopy({
        "verification_date": max((dates.get(path, "2026-10-02") for path in selected),
                                 default="2026-10-02"),
        "verification_scope": "upstream byte identity and declared benchmark structure; not generator reproduction",
        "asset_sha256": {path: item[3] for path, item in selected.items()},
        "upstream_assets": {path: {"repository": item[0], "revision": item[1], "path": item[2]}
                            for path, item in selected.items()},
        "license_status": "unknown; code licensing does not establish data redistribution rights",
        "units": "not independently verified; compare fixed benchmark observations only",
    })
