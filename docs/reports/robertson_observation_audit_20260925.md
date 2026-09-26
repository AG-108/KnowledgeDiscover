# Robertson nonuniform observation and derivative audit (2026-09-25)

The current generated Robertson data use Radau and a logarithmic observation
grid from 0 to 100. The benchmark's ODE bridge calls `np.gradient(state, t)`
with the actual nonuniform time coordinates, then removes the first and last
derivative estimate from each trajectory. This server-side audit compared those
interior estimates with the analytic RHS at the corresponding **clean** states.
It used five generator seeds, six trajectories per seed, the standard 241-point
grid, strides 1 and 2, and either no observation noise or 1% of each
trajectory/state's standard deviation. This is a data-quality diagnostic, not
a model fit or a proposal to supply the analytic RHS to a discovery method.

The first positive observation time is `1e-6`; the smallest spacing is
`8.01e-8`, the largest is `7.42`, and their ratio is about `9.26e7`. As a
check that the code handles nonuniform spacing, `np.gradient(t**2, t)` matches
the analytic interior derivative `2*t` to a maximum absolute error of
`8.53e-14` on this grid.

The table reports relative RMS derivative error for the first (`x`) and third
(`z`) Robertson targets, pooled across seeds and trajectories. The denominator
is the RMS of the analytic clean derivative on the same retained points.

| Time stride | State noise | All interior points: x / z | Score only at `t >= 0.1`: x / z | Score only at `t >= 1`: x / z |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0% | 0.0001 / 0.0004 | 0.0003 / 0.0003 | 0.0006 / 0.0006 |
| 1 | 1% | 50,061 / 64,390 | 0.953 / 1.010 | 0.201 / 0.207 |
| 2 | 0% | 0.0005 / 0.0017 | 0.0012 / 0.0012 | 0.0025 / 0.0025 |
| 2 | 1% | 24,164 / 34,295 | 0.509 / 0.466 | 0.117 / 0.101 |

For stride 1, the `t >= 1` diagnostic retains 1,770 of 7,170 interior
points per target across the 30 generated trajectories. Even after discarding
roughly three quarters of the interior observations, 1% state noise still
produces around 20% relative derivative error for x and z. The result is
consistent with noise divided by extremely small early time steps, rather
than a uniform-step implementation mistake. A time cutoff changes the data
distribution and is **not** an adopted noise protocol.

**Decision:** Keep clean Robertson results separate from noisy/sparse stress
results. Do not promote the current 1% noisy Robertson derivative targets into
a formal recovery track. Before doing so, predeclare an observation schedule
and derivative estimator, validate their error by component and time region,
and retain all failed/nonfinite targets in the denominator. An analytic RHS
target is suitable for this audit only; using it for discovery would reveal
the synthetic ground truth.

Raw per-cutoff metrics and the exact grid diagnostics are in
`results/server_robertson_grid_audit_20260925/audit.json`; the script is
`results/server_robertson_grid_audit_20260925/audit.py`.
