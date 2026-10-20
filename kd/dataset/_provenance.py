"""Source evidence declarations; unknown units never become inferred truth."""

from copy import deepcopy

SOLID_REVISION = "b4af0b5b49a1c1d0392a6504ec12a2c763136957"
SOLID_SOURCE = "https://github.com/woshixuhao/Discovery_of_soild_consititutive/blob/" + SOLID_REVISION
FIELD_EVIDENCE = {
    "heat": {"axis_evidence": "embedded x,t; usol[x,t]",
             "equation_evidence": None, "boundary": "unknown", "units": {"x": "unknown", "t": "unknown", "u": "unknown"},
             "missing_evidence": ["generator parameters including diffusivity", "boundary and physical units"]},
    "burgers_sine": {"axis_evidence": "EqGPT/code/surrogate_model.py: Burgers_equation; embedded x,t",
                     "equation_evidence": "EqGPT/data/Burgers_equation/dataset_description",
                     "boundary": "unknown", "units": {"x": "unknown", "t": "unknown", "u": "unknown"},
                     "missing_evidence": ["generator boundary", "physical units"]},
    "allen_cahn": {"axis_evidence": "EqGPT/code/surrogate_model.py: Allen_Cahn; embedded x,t",
                   "equation_evidence": "EqGPT/data/Allen_Cahn/data_description",
                   "boundary": "unknown", "units": {"x": "unknown", "t": "unknown", "u": "unknown"},
                   "missing_evidence": ["generator boundary", "physical units"]},
    "convection_diffusion": {"axis_evidence": "EqGPT/code/surrogate_model.py: Convection_diffusion_equation; u.T",
                             "equation_evidence": "EqGPT/data/Convection_diffusion_equation/dataset_description",
                             "boundary": "unknown", "units": {"x": "unknown", "t": "unknown", "u": "unknown"},
                             "missing_evidence": ["generator boundary", "physical units"]},
}

PARAMETRIC_PAPER = "https://arxiv.org/pdf/2005.07916"
CHECKED_EVOLUTION_EVIDENCE = {
    "parametric_burgers": {
        "source_status": "paper_and_observation_consistent; upstream_bytes_unverified",
        "axis_evidence": "paper III.A.1: u[t,x]; 200 times [0,1), 251 positions [0,1]",
        "equation_evidence": PARAMETRIC_PAPER + "#page=12",
        "initial_condition": "(1.5-x)*sin(2*pi*x)",
        "boundary": {"kind": "dirichlet", "verified": True, "value": 0.0,
                     "axes": ["x"], "evidence": PARAMETRIC_PAPER + "#page=12"},
        "missing_evidence": ["byte-identical upstream Burgers file", "physical units"],
    },
    "parametric_convection_diffusion": {
        "source_status": "upstream_byte_verified",
        "axis_evidence": "paper III.A.2: u[t,x]; 250 times [0,5), 201 positions [0,8]",
        "equation_evidence": PARAMETRIC_PAPER + "#page=14",
        "initial_condition": "(8-x)*sin(x)",
        "boundary": {"kind": "dirichlet", "verified": True, "value": 0.0,
                     "axes": ["x"], "evidence": PARAMETRIC_PAPER + "#page=14"},
        "missing_evidence": ["physical units"],
    },
    "advection_diffusion": {
        "source_status": "upstream_byte_verified",
        "axis_evidence": "Expression1 C-order reshape(51,51,61,4); channels x,y,t,u",
        "layout_evidence": "https://github.com/PhIMaL/DeePyMoD/blob/"
                           "543c2e5fc7a6a5637be660775873724d884790ba/"
                           "examples/PDE_2D_Advection-Diffusion.ipynb",
        "equation_evidence": None,
        "boundary": "unknown; opposite observed edges disagree; no periodic wrapping",
        "missing_evidence": ["generating coefficients", "boundary conditions", "physical units"],
    },
}

# A train rate must vary to identify a rate-dependent law; this is an eligibility
# rule shared by all methods, not a choice made from a method's test score.
CONSTITUTIVE_ELIGIBILITY = {
    "min_train_rate_groups": 2, "min_test_rate_groups": 1,
    "min_train_samples": 2, "min_test_samples": 1,
    "constant_training_target": "not_applicable",
    "rate_equality": "exact_float64; no evidence-supported tolerance",
}


def source_evidence(name):
    """JSON-safe declarations with explicit gaps, usable without external data."""
    if name in CHECKED_EVOLUTION_EVIDENCE:
        from ._advanced import EVOLUTION_SPECS
        from ._core_sources import evidence

        spec = EVOLUTION_SPECS[name]
        result = deepcopy(CHECKED_EVOLUTION_EVIDENCE[name])
        result.update(evidence([spec["path"]]))
        result.update(
            verification_date="2026-10-16",
            verification_scope=("upstream byte identity and source layout" if
                                result["source_status"] == "upstream_byte_verified" else
                                "local SHA256, paper layout, initial/boundary values and numerical residual"),
            asset_sha256={spec["path"]: spec["sha256"]},
            reference_status="paper_declared_not_certified" if spec["reference"] else "unknown",
            generalization="same-trajectory temporal block residual; no forward rollout",
        )
        return result
    if name.startswith("fde_"):
        from ._fractional import FDE_PDE_SPECS, FDE_SPECS, pde_evidence, response_evidence
        if name in FDE_PDE_SPECS:
            return pde_evidence(name)
        if name in FDE_SPECS:
            return response_evidence(name)
    if name in FIELD_EVIDENCE:
        return deepcopy(FIELD_EVIDENCE[name])
    if name.startswith("cyt_"):
        from ._base import _CYT_COLUMNS

        return {
            "units": {column: "unknown" for column in _CYT_COLUMNS},
            "column_evidence": "_base._CYT_COLUMNS; FlowFeature.dat header",
            "scaling": "identity; generator nondimensionalization unknown",
            "feature_generation": "missing solver inputs and feature export code",
            "spatial_gap": None,
            "missing_evidence": ["CFD scales and export formulas", "feature stencil support and boundary neighborhoods",
                                 "independent feature availability", "cross-geometry comparability"],
            "generalization": "within_case_spatial_holdout; cross_geometry_OOD_not_verified",
        }
    if name.startswith("solid_"):
        return {
            "units": {"strain": "dimensionless by quantity definition",
                      "strain_rate": "s^-1; upstream pack_data plot label",
                      "stress": "unknown source stress unit", "DIF": "dimensionless ratio by definition"},
            "scaling": "identity; pack_data copies strain/stress without rescaling",
            "source_revision": SOLID_REVISION,
            "scaling_evidence": SOLID_SOURCE + "/codes/Discovery_of_hardening_strain_rate.py#L531",
            "DIF_evidence": SOLID_SOURCE + "/codes/Discovery_of_hardening_strain_rate.py#L511",
            "DIF_matching": "exact rate dictionary lookup; missing rate defaults to 1.0 upstream",
            "material_id": None,
            "material_study_id": None,
            "missing_evidence": ["per-study material/heat/loading identifiers", "original stress units",
                                 "cross-study material equivalence"],
            "eligibility": deepcopy(CONSTITUTIVE_ELIGIBILITY),
        }
    if name.startswith("rubber_"):
        return {
            "units": {"lambda": "dimensionless; 1+nominal strain", "C": "unknown compound code",
                      "T": "unknown; filename token does not establish K or Celsius", "stress": "unknown"},
            "scaling": "lambda=1+nominal strain; others identity",
            "split_evidence": "data_rubber/train/readme.md and test/readme.md",
            "generalization": "official holdout; shared conditions audited separately",
            "missing_evidence": ["stress/temperature units", "compound code definition", "official sampling rationale"],
        }
    if name == "ball_drop":
        return {"units": "loader conversions in _base.load_ball_drop_dataset",
                "split": "whole trajectory", "truth": None,
                "missing_evidence": ["per-trajectory measurement provenance; no exact dynamical truth"],
                "statistical_unit": "physical ball/system, not derivative RHS"}
    if name.startswith("vgs_"):
        return {"units": "physical units unverified; loader dx=.02 and case-specific t_coeff",
                "scaling": "raw height; upstream surrogate division by 50 is not applied",
                "statistical_unit": name.split("_")[1],
                "window": name.split("_", 2)[2], "window_time": "reset to zero",
                "missing_evidence": ["simulation geometry/scales/boundary", "absolute window time origin"]}
    if name == "wave_breaking":
        return {"units": "unknown; source t,x,elevation columns",
                "projection": "frame-local linear spatial interpolation; no time interpolation or smoothing",
                "grid_geometry": "intersection of coordinate extents across frames; no response-based fitting",
                "statistical_unit": "wave record; independence of 23 records unverified",
                "missing_evidence": ["experiment scales/geometry/boundary", "physical record independence"]}
    if name.startswith("tlc_"):
        return {"units": "unknown; CSV columns declare coordinates/channels",
                "equation_evidence": None, "boundary": "unknown",
                "missing_evidence": ["generator equations/coefficients and boundaries", "physical units"]}
    return {}
