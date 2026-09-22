"""Dataset-driven usage of KD_DSCV (Mode 1 / regular grids)."""

from _common import bootstrap_project_root

project_root = bootstrap_project_root()

from kd.dataset import load_pde_grid
from kd.model.kd_dscv import KD_DSCV

# Load the PDE dataset through the public registry.
pde_dataset = load_pde_grid("chafee-infante")

# Configure the DISCOVER model.
model = KD_DSCV(
    n_iterations=20,
    n_samples_per_batch=200,
    binary_operators=["add", "mul", "diff", "diff2"],
    unary_operators=["n2"],
)

# Pass the GridPDEDataset directly to the model.
model.import_dataset(pde_dataset)

# Keep the example short; increase n_iterations for real experiments.
result = model.train(n_epochs=10, verbose=False)
print(f"Current best expression: {result['expression']}")
