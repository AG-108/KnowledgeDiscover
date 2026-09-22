# Physical credibility: agreed evaluation target

User decision: assess an equation's physical credibility. Do not score scientific
explanatory value, novelty, conformity to familiar equations, or narrative appeal.
This is a design specification; no human agreement or scientific validation results
have been measured yet.

## What a result means

An evaluation reports consistency with observed evidence and explicitly declared
physical assumptions on a specified domain. Passing checks is not proof that an
equation is the true governing law, or that the available observations identify it
uniquely. Report non-identifiability and unknown checks explicitly.

## Constraint contract

Each check has a stable identifier and records:

- The property under test, source, and source version.
- Whether it is supplied by the problem, mathematically derived, or a proposed
  hypothesis. Proposed hypotheses are diagnostic and cannot silently become gates.
- Applicable variables, units, physical regime, domain, and boundary conditions.
- A candidate-independent test definition, numerical tolerances, and test budget.
- One of pass, fail, unknown, not_applicable, or evaluation_error.
- Reproducible evidence and a concise explanation grounded in that evidence.

The dataset card and check definitions are fixed before viewing method identities
or candidate equations. The same tests apply to all eligible methods. Ground truth
can validate the evaluator offline but is withheld from the deployed LLM judge.

## Check families and limits

| Family | Evidence | Important applicability condition |
| --- | --- | --- |
| Dimensional consistency | Unit algebra / nondimensionalization map | Units and coefficient dimensions must be known |
| Domain validity | Symbolic domain analysis and fixed-domain numerical probes | Physical domain must be declared independently of candidates |
| Conservation or balance | Derived integral identity and numerical balance residual | Include source terms and boundary fluxes; closed-system conservation is not universal |
| Symmetry | Symbolic transformation or controlled numerical comparison | The symmetry must be justified for this problem |
| Sign or dissipation | Energy/balance argument or specified parameter constraints | Applicable physical regime and boundary conditions must be documented |
| Limiting behavior | Independently specified limits and numerical/asymptotic checks | A limit outside the model's validity domain cannot be imposed as a gate |
| Empirical compatibility | Held-out observations and preregistered trajectory checks | Keep data fit separate from physical-constraint consistency |

Numerical simulation failure needs diagnosis: discretization error, insufficient
resolution, and integration failure must not automatically count as a physical
contradiction. Validate test machinery against a known appropriate reference case.

## LLM role

The LLM may help draft checks from the problem description and interpret verified
evidence. A reviewed checklist, symbolic calculations, and reproducible numerical
tests provide the score's factual basis. The LLM cannot override measured evidence
or insert an unstated textbook equation as the expected answer.

Persist judge model/version, prompt/version, equation normalization, tool results,
declared assumptions, repeated judgments and disagreement/abstention. Do not show
baseline identity or a candidate author's persuasive explanatory text to the judge.

## Scoring and validation

Initially report a vector of check outcomes, violations, coverage and empirical
performance. Do not reward missing evidence through a pass-only denominator or
penalize unknowns as proven physical violations. A future composite needs an
independent validation set, predetermined weights and sensitivity analysis.

Create candidate-pair challenges for equivalent rewrites, wrong signs, missing and
redundant terms, invalid domains and unjustified physical assumptions. Expert labels
must be collected, not inferred from a generator's intended mutation. Include an
unknown/ambiguous option and preserve disagreements.

Compare numeric-only, direct-LLM, and evidence-assisted evaluation. Measure expert
agreement, violation detection, unsupported-claim rate, abstention/coverage, and
sensitivity to candidate order, variable renaming and algebraically equivalent form.

This prototype must operate offline with an injected/mock judge. Paid service calls
and benchmark-wide scoring require a separately configured evaluation run.
