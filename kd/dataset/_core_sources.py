"""Fixed upstream identities checked byte-for-byte on 2026-10-02.

This establishes file lineage, not new physical-unit or redistribution claims.
"""

from copy import deepcopy

DISCOVER = "https://github.com/menggedu/DISCOVER"
DISCOVER_REVISION = "6bf37ce110ccbcb07fb14108cc46b185cdf8fc84"
DSO = "https://github.com/dso-org/deep-symbolic-optimization"
DSO_REVISION = "8348d5b08d1eef6170fdbfd222a492ded990ea12"
ASSETS = {
    "data/benchmarks.csv": (DSO, DSO_REVISION, "dso/dso/task/regression/benchmarks.csv",
                            "42525c6795f3ebc7a6f38d59dbbb011ce92e0ef7d0b533346be8fc0e29abad9a"),
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
    return deepcopy({
        "verification_date": "2026-10-02",
        "verification_scope": "upstream byte identity and declared benchmark structure; not generator reproduction",
        "asset_sha256": {path: item[3] for path, item in selected.items()},
        "upstream_assets": {path: {"repository": item[0], "revision": item[1], "path": item[2]}
                            for path, item in selected.items()},
        "license_status": "unknown; code licensing does not establish data redistribution rights",
        "units": "not independently verified; compare fixed benchmark observations only",
    })
