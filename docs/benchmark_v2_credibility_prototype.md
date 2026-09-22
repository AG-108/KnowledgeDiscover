# Physical-credibility prototype

This is an offline, unvalidated prototype. It reports consistency with a reviewed,
candidate-independent dataset card, not equation truth, explanatory value, novelty,
or familiarity. Its result is a vector plus status coverage; there is no composite.

The bounded implementation supports dimensional homogeneity, fixed-domain real/finite
probes, and explicitly declared sign constraints. Domain and sign checks select a
fixed `lhs`, `rhs`, or `residual` target; explicit-law cards should normally select
`rhs`. Every check records applicability,
assumptions, provenance, tolerance, and budget. Missing units—including unknown
coefficient units—produce `unknown`; known contradictions such as adding length and
time produce `fail`. Bare numeric literals other than structural -1/0/1 have unknown
coefficient units by default. A card may declare `numeric_literal_policy="dimensionless"`
only when its independently reviewed nondimensionalization warrants that assumption.
Absent assumptions produce `not_applicable`;
evaluator exceptions produce `evaluation_error`, never a physical failure. Numerical
probes are budget-bounded, diagnostic, report their discrete coverage, and are not
proofs over continuous domains. Unsimplified parse trees are retained so removable
singularities such as `x/x` at zero are still detected by validity probes.

The optional judge is dependency-injected and offline. It receives a frozen, hashed
payload containing the normalized equation and evidence, but no method identity or
ground truth. Responses containing protected evidence/status fields are rejected.
Its payload now includes the declared physical setting, units, assumptions and
frozen check definitions; descriptive dataset identifiers are omitted to reduce
equation-name cues. Mathematical candidate text uses an AST allowlist, not Python
evaluation. Only bounded scalar arithmetic and the documented elementary functions
are accepted. Invalid syntax returns evaluator-error evidence.

Run the challenge workflow with:

```powershell
python scripts/generate_physical_credibility_challenges.py --output-dir artifacts/credibility
```

The generated mutation intent is not a label. The CSV is deliberately blank and allows
`a`, `b`, `equivalent`, `unknown`, and disagreement-preserving expert judgments. Future
validation should randomize pair order and aliases, then compare numeric-only,
direct-LLM, and evidence-assisted conditions on expert agreement, violation detection,
unsupported-claim rate, abstention/coverage, and order/renaming/equivalence sensitivity.
