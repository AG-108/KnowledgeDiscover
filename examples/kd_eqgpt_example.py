"""Minimal example: EqGPT baseline discovering the KdV equation.

EqGPT pretrains a GPT on a 221-equation math-handbook corpus (excluding the
target equation), fits a surrogate network to the KdV solution data, then
searches for a PDE structure by sampling candidate term sequences from the
GPT and scoring them via sparse regression over the surrogate's autograd
derivatives (see kd/model/eqgpt/ and kd/model/kd_eqgpt.py for the adapted
implementation; the original repo is vendored at kd/dataset/EqGPT/).

Settings below are intentionally small (few pretraining epochs, a coarse
meta-data grid, few search samples) to keep this example fast (~2 minutes
on CPU) -- this trades off reliability for speed, so the final answer is not
guaranteed to be the textbook KdV form. Rewards do not necessarily improve
monotonically across search epochs either (a higher-reward, spurious fit can
appear later), so the printed history is worth reading in full, not just the
last row.
"""

from _common import bootstrap_project_root

project_root = bootstrap_project_root()

from kd.dataset import load_kdv_equation
from kd.model.kd_eqgpt import KD_EqGPT

# Load the same KdV dataset used by the other PDE baseline examples.
dataset = load_kdv_equation()

# Use a small EqGPT configuration for a quick demonstration.
model = KD_EqGPT(
    gpt_pretrain_epochs=5,
    augment_times=8,
    surrogate_iters=1000,
    choose=2000,
    choose_validate=500,
    optimize_epochs=3,
    samples=50,
    meta_nx=20,
    meta_nt=20,
    device="cpu",
    verbose=True,
    seed=0,
)
model.fit(dataset, equation_name="kdv")

# Display the search history and final result.
print("\nSearch history (per epoch, best-of-elite):")
for entry in model.history_:
    print(f"  epoch {entry['epoch']}: {entry['best_equation']}  (reward={entry['best_award']:.4f})")

print(f"\nFinal discovered equation: {model.best_pde_}")
print(f"Final reward: {model.best_award_:.4f}")
print("Reference (true KdV form): ut+u*ux+uxxx = 0")
