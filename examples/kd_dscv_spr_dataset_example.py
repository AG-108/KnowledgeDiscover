"""Dataset-driven usage of KD_DSCV_SPR (Mode 2 / sparse PINN pipeline)."""

from _common import bootstrap_project_root

project_root = bootstrap_project_root()

from kd.dataset import load_pde_grid
from kd.model.kd_dscv import KD_DSCV_SPR

# Load the Burgers dataset for this example.
pde_dataset = load_pde_grid("burgers")

# Configure the R-DISCOVER model.
model = KD_DSCV_SPR(
    n_iterations=5,
    n_samples_per_batch=50,
    binary_operators=["add_t", "mul_t", "div_t", "diff_t", "diff2_t"],
    unary_operators=["n2_t"],
)

# Import the grid directly and set random_state for reproducible sampling.
model.import_dataset(
    pde_dataset,
    sample_ratio=0.05,
    colloc_num=256,
    random_state=0,
)

# Run one demonstration iteration; production runs should use a larger budget.
step_result = model.train(n_epochs=1, verbose=False)
print(f"Current reward snapshot: {step_result['r']}")
