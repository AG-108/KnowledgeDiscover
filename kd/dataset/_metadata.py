"""Scientific task cards, distinct from numerical SR/ODE/PDE dispatch.

No data reads or model imports: pending families stay visible in the coverage
inventory even in a public checkout without optional assets.
"""

from copy import deepcopy

DATASET_PROTOCOL_VERSION = "domain-v4"


def dataset_metadata(name, task, *, family=None):
    """Return JSON-safe protocol declarations; availability is checked separately."""
    from ._domain import FIELD_SPECS, SOLID_SUBDIRS

    card = {
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "scientific_task": {"sr": "algebraic_formula", "ode": "ode_dynamics",
                            "pde": "evolution_pde", "steady_pde": "steady_pde", "green":"green_operator_inverse"}[task],
        "physical_family": family if family not in {None, "sr", "ode", "pde", "steady_pde", "regression"} else name,
        "target_definition": {"sr": "y=f(features)", "ode": "state_time_derivatives",
                              "pde": "u_t=F(field,spatial_derivatives,coordinates)", "steady_pde": "F(fields,spatial_derivatives)=0", "green":"U=K@(w*F)+U_hom"}[task],
        "split_protocol": {"sr": "provided_benchmark_split", "ode": "trajectory_holdout",
                           "pde": "temporal_block_holdout", "steady_pde": "spatial_blocks_with_support_gap", "green":"whole_forcing_60_20_20"}[task],
        "instance_policy": "single_dataset",
        "benchmark_role": "extension_candidate" if task == "pde" else "candidate",
        "reference_status": "unknown",
        "source_paths": [],
        "source_status": "provenance_pending",
        "integration_status": "adapter_available; formal_protocol_not_frozen",
        "requirements": [],
    }
    from ._advanced import EVOLUTION_SPECS, STEADY_SPECS
    advanced = {**EVOLUTION_SPECS, **STEADY_SPECS}
    if name in advanced:
        spec = advanced[name]
        card.update(source_paths=[spec["path"]], physical_family="steady_fields" if task == "steady_pde" else name,
                    source_status="packaged_observation_and_description",
                    reference_status="source_declared_not_certified" if spec.get("reference") else "unknown",
                    integration_status="equation_adapter_available; source_limits_recorded",
                    requirements=["source_geometry"] if spec.get("geometry_status") == "source_blocked" else [],
                    benchmark_role="physical_extension")
    elif task == "green":
        from ._green import GREEN_SPECS
        spec = GREEN_SPECS[name]
        card.update(source_paths=[spec["path"]], physical_family=spec["family"],
                    source_status="local_observed_forcing_response", reference_status="source_declared_or_missing; not_certified",
                    integration_status="integral_loss_adapter_available",
                    instance_policy="one_operator_kernel", benchmark_role=spec["variant"],
                    requirements=["source_units_boundary_reference_and_noise_scale"])
    elif name.startswith("dso_file_"):
        from ._file_sr import PREFIX
        label=name.removeprefix("dso_file_")
        paths=[PREFIX+label+".csv"]
        if label!="PiecewiseFunction-1":
            paths.append(PREFIX+label+"_test.csv")
        card.update(source_paths=paths, source_status="local_headerless_CSV",
                    reference_status="unknown", instance_policy="official_pair" if len(paths)==2 else "single_file_input_holdout",
                    split_protocol="official_file_train_test_pair" if len(paths)==2 else "input_rank_middle_fifth_holdout",
                    integration_status="explicit_file_adapter_available", target_definition="last_numeric_column",
                    physical_family="DSO_file_"+label.split("-")[0], requirements=["source_formula_units_license"])
    elif name == "pyqg_subgrid":
        from ._closure import PYQG_PATHS
        card.update(scientific_task="turbulence_closure",physical_family="pyqg",source_paths=PYQG_PATHS,
                    target_definition="q_subgrid_forcing per layer from resolved fields only",
                    split_protocol="run0_train_run1_test; internal_temporal_gap2",
                    source_status="NetCDF_variable_and_coarsening_metadata",
                    integration_status="offline_resolved_field_adapter_available",
                    instance_policy="one_two_layer_system; two targets",benchmark_role="physical_extension",
                    requirements=["generator_version_filter_cutoff_license; online_validation_future"])
    elif task == "sr":
        card.update(source_paths=["data/benchmarks.csv"], source_status="packaged_definition",
                    reference_status="formula_declared", physical_family=name.rsplit("-", 1)[0])
    if name in advanced:
        pass
    elif name.startswith("ode_core_"):
        development = {"oscillator", "population", "chaotic", "brusselator"}
        card.update(source_paths=["_ode_core.py"], source_status="local_generator",
                    reference_status="generator_declared", instance_policy="multi_trajectory_system",
                    benchmark_role="development" if name[9:] in development else "core_main")
    elif name == "ball_drop":
        card.update(source_paths=["discovery-of-physics-from-data/data/Ball_drops_data.xls",
                                  "discovery-of-physics-from-data/data/balls.txt"],
                    source_status="local_research_snapshot", benchmark_role="physical_extension",
                    target_definition="d(h)/dt,d(v)/dt conditioned on mass,diameter",
                    instance_policy="multi_trajectory_system", requirements=["trajectory_reference_validation"])
    elif name.startswith("cyt_"):
        from ._base import _CYT_CASE_ALIASES

        card.update(scientific_task="turbulence_closure", physical_family="CYT",
                    target_definition="Mut=f(U,V,Ru,P,Ux,Uy,Vx,Vy,Px,Py,T,dis,Ma,AoA,Re)",
                    split_protocol="spatial_block_holdout", reference_status="unknown",
                    source_status="provenance_pending", benchmark_role="physical_extension",
                    source_paths=[f"CYT/{_CYT_CASE_ALIASES[name[4:]]}/Output/FlowFeature.dat"],
                    requirements=["unit_and_source_verification", "cross_case_validation"],
                    integration_status="within_case_spatial_protocol; cross_case_OOD_not_established")
    elif name in SOLID_SUBDIRS:
        subdir, pattern = SOLID_SUBDIRS[name]
        card.update(scientific_task="constitutive_law", physical_family="solid_constitutive",
                    target_definition="DIF=f(strain_rate)" if name == "solid_dif" else
                                      "stress=f(strain,strain_rate); no DIF predictor",
                    split_protocol="within_curve_rate_holdout" if name == "solid_dif" else
                                   "within_material_rate_holdout",
                    instance_policy="source_curve" if name == "solid_dif" else "source_material_study",
                    reference_status="unknown", source_status="local_research_snapshot",
                    source_paths=[f"Discovery_of_soild_consititutive/data/{subdir}/{pattern}"],
                    benchmark_role="physical_extension",
                    requirements=["cross_study_material_identity", "source_unit_verification"],
                    integration_status="source_instances_available; cross_study_identity_pending")
    elif name.startswith("rubber_"):
        card.update(scientific_task="constitutive_law", physical_family="rubber",
                    target_definition="stress=f(lambda,C,T)", split_protocol="official_rubber_split",
                    instance_policy="official_train_test_pair", reference_status="unknown",
                    source_status="local_research_snapshot", benchmark_role="physical_extension",
                    source_paths=[f"Discovery_of_soild_consititutive/data/data_rubber/{name[7:]}/*.xlsx"])
    elif name.startswith("tlc_"):
        from ._catalog import _TLC_TIME_SERIES_FILES

        regular = {"tlc_burgers1d", "tlc_heat_longtime", "tlc_heat_multiscale_lesspoints"}
        card.update(source_paths=["TLC/" + _TLC_TIME_SERIES_FILES[name]],
                    physical_family="TLC_" + name.split("_")[1], benchmark_role="physical_extension")
        if name != "tlc_burgers1d":
            card.update(integration_status="equation_adapter_available_or_source_blocked", requirements=[
                "multidimensional_or_coupled_equations",
                "anisotropic_grid_adapter" if name in regular else "scatter_derivative_adapter"])
        if name == "tlc_ns_long":
            card["target_definition"] = "coupled u,v momentum and pressure/continuity constraints"
        elif name == "tlc_wave_darcy":
            card["target_definition"] = "wave PDE; temporal order and coefficients require source verification"
    elif name == "wdwake":
        card.update(physical_family="wake", target_definition="coupled two_velocity_field_PDE",
                    source_paths=["WDwake/TI8_U.npy", "WDwake/TI8_V.npy"],
                    benchmark_role="physical_extension", integration_status="blocked_physical_protocol",
                    requirements=["physical_coordinates_units", "anisotropic_spacing", "boundary_conditions"])
    elif name.startswith("vgs_"):
        card.update(physical_family="VGS_" + name.split("_")[1], target_definition="h_t=F(h,derivatives)",
                    source_paths=[f"ViscousGravityCurrent/data/{name}.dat"], benchmark_role="physical_extension",
                    requirements=["same_family_window_split", "boundary_and_unit_verification"])
    elif name == "wave_breaking":
        card.update(physical_family="wave_breaking", source_paths=["WaveBreaking.pkl"],
                    instance_policy="wave_record", benchmark_role="physical_extension",
                    target_definition="wave_elevation_time_derivative",
                    requirements=["projection_error", "source_geometry_and_boundary"])
    elif name == "pdeformer_sinus":
        card.update(physical_family="pdeformer_sinus", source_paths=["_pdeformer_sinus.py"],
                    source_status="local_generator", instance_policy="generated_equation",
                    reference_status="generator_declared", requirements=["variable_coefficient_library"])
    elif name in FIELD_SPECS:
        spec = FIELD_SPECS[name]
        card.update(physical_family=spec["family"], source_paths=[spec["path"]],
                    source_status="local_field_and_description" if spec["rhs"] else "embedded_coordinates_only",
                    reference_status="source_declared" if spec["rhs"] else "unknown",
                    requirements=["source_provenance", "boundary_verification"])
    elif task == "pde":
        from ._registry import PDE_REGISTRY

        spec = PDE_REGISTRY.get(name, {})
        card["source_paths"] = ["data/" + p for p in
                                (spec.get("files", {}).values() if "files" in spec else [spec["file"]])]
        if name in {"burgers", "kdv", "chafee-infante"}:
            card.update(benchmark_role="core_main", reference_status="coefficient_declared")
        elif name in {"fisher", "fisher_linear"}:
            card["benchmark_role"] = "development"
    if name in EVOLUTION_SPECS:
        spec = EVOLUTION_SPECS[name]
        card["target_definition"] = (
            "implicit relation with u_t,u_xt,u_x" if name == "implicit_mixed" else
            "u_"+"t"*spec["order"]+"=F(observed fields,spatial derivatives)")
    elif name.startswith("tlc_") and name not in advanced:
        regular = {"tlc_burgers1d", "tlc_heat_longtime", "tlc_heat_multiscale_lesspoints"}
        card.update(integration_status="observed_residual_available" if name in regular else "source_blocked",
                    requirements=["boundary_and_units"] if name in regular else ["geometry_and_interface_evidence"])
    from ._provenance import source_evidence

    card["source_evidence"] = source_evidence(name)
    return deepcopy(card)


def pending_datasets():
    """Non-executable local candidates and known aliases, not scheduler entries."""
    from ._catalog import TLC_UNSUPPORTED_STEADY_STATE

    rows = []

    def add(name, physical, paths, reason, *, family=None, role="extension_candidate", aliases=None):
        rows.append({"dataset": name, "numerical_task": None, "scientific_task": physical,
                     "physical_family": family or name, "source_paths": paths,
                     "integration_status": "not_integrated", "requirements": [reason],
                     "benchmark_role": role, "aliases": aliases or [],
                     "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
                     "reference_status": "not_verified", "source_status": "provenance_pending",
                     "instance_policy": "requires_definition", "split_protocol": "requires_definition",
                     "target_definition": reason})

    for path in TLC_UNSUPPORTED_STEADY_STATE:
        add("tlc_steady_" + path.split("/")[-1][:-4], "steady_pde", ["TLC/" + path],
            "steady residual, source/boundary and geometry split", family="TLC_steady")
    prefix = "Discover_Green_function/dso/task/pde/"
    for family in ("advection", "helmholtz", "negative_helmholtz", "laplace", "periodic_helmholtz", "jump_green"):
        folder = "data2" if family == "jump_green" else "data"
        add("green_" + family, "green_operator_inverse", [prefix + folder + "/" + family + ".mat"],
            "forcing-response column split and integral forward loss", family="green_" + family)
        if family != "advection":
            for noise in (1, 5, 10):
                add(f"green_{family}_noise_{noise}", "green_operator_inverse",
                    [f"{prefix}data2/{family}_noise_{noise}.mat"],
                    "noise protocol and integral forward loss", family="green_" + family, role="noise_variant")
    for k in (6, 9):
        add(f"green_periodic_helmholtz_k{k}", "green_operator_inverse",
            [f"{prefix}data/periodic_helmholtz_k={k}.mat"], "parameter holdout and integral forward loss",
            family="green_periodic_helmholtz", role="parameter_variant")
    for family in ("negative_helmholtz", "periodic_helmholtz"):
        add("green_" + family + "_data2", "green_operator_inverse", [prefix + f"data2/{family}.mat"],
            "different bytes from data version; source identity unresolved", family="green_" + family,
            role="unresolved_variant")
    add("pyqg_subgrid", "turbulence_closure", [prefix + "data/eddy_forcing1_run=*_scipy.nc"],
        "field closure targets, filtering and run holdout", family="pyqg")
    fields = [
        ("beam", "data/Beam_equation.mat", "evolution_pde", "verify temporal/spatial derivative orders"),
        ("klein_gordon", "data/KG_equation.mat", "evolution_pde", "second temporal derivative adapter"),
        ("wave", "EqGPT/data/Wave_equation/wave.mat", "evolution_pde", "second temporal derivative adapter"),
        ("burgers2d", "EqGPT/data/Burgers_2D/Burgers2D.mat", "evolution_pde", "2D scalar PDE adapter"),
        ("implicit_mixed", "EqGPT/data/Eq_6_2_12/data_Eq_6_2_12.csv", "evolution_pde", "mixed u_xt implicit equation"),
        ("pde_compound_eqgpt", "EqGPT/data/PDE_compound/PDE_compound.csv", "evolution_pde", "source sign conflict and high reference residual"),
        ("parametric_burgers", "data/Parametric_Burgers_equation.npy", "evolution_pde", "coordinates and variable coefficients missing"),
        ("parametric_convection_diffusion", "data/Parametric_convection_diffusion.npy", "evolution_pde", "coordinates and variable coefficients missing"),
        ("advection_diffusion", "data/Advection_diffusion.mat", "evolution_pde", "Expression1 layout undecoded"),
        ("laplace_disk", "EqGPT/data/Possion_equation/Possion_x_y.xlsx", "steady_pde", "steady Laplace residual and disk geometry"),
        ("laplace_smile", "EqGPT/data/Laplacian_smile/Laplacian_smile.xlsx", "steady_pde", "steady Laplace residual and mask"),
        ("poisson_eitech", "EqGPT/data/Laplacian_EITech/Laplacian_EITech.xlsx", "steady_pde", "steady Poisson residual and source"),
    ]
    for name, path, physical, reason in fields:
        add(name, physical, [path], reason)
    dso = "DeepSymbolicOptimization/dso/dso/task/regression/data/"
    for name in [f"Custom-{i}" for i in range(1, 24)] + ["Test-1", "PiecewiseFunction-1"]:
        paths = [dso + name + ".csv"]
        if name != "PiecewiseFunction-1":
            paths.append(dso + name + "_test.csv")
        add("dso_file_" + name, "algebraic_formula", paths, "verify column semantics and reference",
            family="DSO_file_" + name.split("-")[0])
    duplicates = {
        "kdv_aliases": ["data/Kdv.mat", "EqGPT/data/KdV_equation/KdV-PINN.mat"],
        "burgers_sine_aliases": ["data/Burgers_equation.mat", "EqGPT/data/Burgers_equation/burgers_sine.mat"],
        "klein_gordon_alias": ["EqGPT/data/KG_equation/KG_Exp.mat"],
        "chafee_aliases": ["EqGPT/data/Chaffee_Infante_equation/CI.npy", "EqGPT/data/Chaffee_Infante_equation/x.npy", "EqGPT/data/Chaffee_Infante_equation/t.npy"],
        "pde_divide_alias": ["EqGPT/data/PDE_divide/PDE_divide.npy"],
        "green_laplace_alias": [prefix + "data2/laplace.mat"],
    }
    for name, paths in duplicates.items():
        add(name, "duplicate_asset", paths, "byte-duplicate assets, excluded from independent task counts",
            role="duplicate")
    add("symbolicgpt_generation_configs", "training_asset", ["SymbolicGPT/datasets/*/config.txt"],
        "configuration only; sample corpus absent", role="training_asset")
    add("eqgpt_equation_handbook", "training_asset", ["EqGPT/code/PDE_dataset_0725.xlsx"],
        "pretraining equations, not held-out field observations", role="training_asset")
    from ._catalog import DATASET_REGISTRY
    return [row for row in rows if row["dataset"] not in DATASET_REGISTRY]
