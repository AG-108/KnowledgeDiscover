"""Evidence-based prior exposure labels; no checkpoint downloads or truth use."""

from copy import deepcopy

# Exact PDE handbook structures are included by current EqGPT pretraining.
# Only kdv has an exclusion mapping; physical observation/checkpoint overlap
# remains a different question from confirmed symbolic structure exposure.
EQGPT_STRUCTURE = {
    "burgers": ["Bateman-Burgers equation", "ut,+,u,*,ux,+,uxx"],
    "burgers_sine": ["Bateman-Burgers equation", "ut,+,u,*,ux,+,uxx"],
    "chafee-infante": ["Chafee-Infante equation (source dash encoding)", "ut,+,uxx,+,u^3,+,u"],
    "allen_cahn": ["Chafee-Infante equation (source dash encoding)", "ut,+,uxx,+,u^3,+,u"],
    "convection_diffusion": ["6.2.13", "ut,+,ux,+,uxx"],
    "burgers2d": ["Burgers_2D", "ut,+,u,*,ux,+,u,*,uy,+,Laplace(u)"],
    "wave": ["(1.2.10)", "utt,+,uxx"],
    "klein_gordon": [
        "Klein-Gordon_u and polynomial variants (source dash encoding)",
        "utt,+,uxx,+,u",
    ],
    "implicit_mixed": ["6.2.12", "ut,+,ux,+,uxt"],
}


def training_overlap(model, dataset):
    if model == "eqgpt":
        evidence = [
            "kd/model/kd_eqgpt.py:_KNOWN_EQUATION_NAMES",
            "kd/model/eqgpt/vocab.py:read_dataset",
            "kd/model/eqgpt/data/PDE_dataset_0725.xlsx",
        ]
        if dataset in EQGPT_STRUCTURE:
            return {
                "status": "confirmed",
                "scope": "symbolic_family_structure_exposure; no coefficient or field-data claim",
                "handbook": deepcopy(EQGPT_STRUCTURE[dataset]),
                "evidence": evidence,
                "heldout_equation_exclusion": False,
                "observation_overlap": "unknown",
                "checkpoint_overlap": "not_applicable; current adapter pretrains per fit",
                "report_partition": "prior_exposed_structure",
            }
        if dataset == "kdv":
            return {
                "status": "possible",
                "scope": "related structures may remain after exact-name exclusion",
                "heldout_equation_exclusion": "KdV equation exact name",
                "evidence": evidence,
                "observation_overlap": "unknown",
                "report_partition": "exposure_audited",
            }
        return {
            "status": "unknown",
            "scope": "family not individually matched to handbook",
            "evidence": evidence,
            "observation_overlap": "unknown",
            "report_partition": "unknown_prior_exposure",
        }
    if model == "e2e":
        return {
            "status": "unknown",
            "scope": "external shared checkpoint training data",
            "evidence": ["kd/model/kd_e2e.py:provenance_ training_data_overlap=unknown"],
            "reason": "No retained corpus/checkpoint lineage establishes disjointness",
            "report_partition": "unknown_prior_exposure",
        }
    if model == "symbolicgpt":
        return {
            "status": "possible",
            "scope": "synthetic grammar permits related algebraic families",
            "evidence": [
                "kd/model/kd_symbolicgpt.py:_build_pretrain_corpus and fit",
                "kd/dataset/SymbolicGPT/datasets/*/config.txt",
            ],
            "checkpoint_overlap": "not_applicable; generated per-fit corpus",
            "real_observation_use_in_pretraining": False,
            "reason": "Configuration/shape/seed-only corpus; exact-family membership not audited",
            "report_partition": "synthetic_prior_exposure",
        }
    return {
        "status": "not_applicable",
        "scope": "no audited pretrained-model corpus for this method",
    }
