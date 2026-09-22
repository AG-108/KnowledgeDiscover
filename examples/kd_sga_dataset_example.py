"""Dataset-driven usage of KD_SGA."""

from _common import bootstrap_project_root

project_root = bootstrap_project_root()

from kd.dataset import load_pde_grid
from kd.model.kd_sga import KD_SGA

# Load the PDE dataset through the public registry.
pde_dataset = load_pde_grid("chafee-infante")

# Configure the model with legacy-compatible parameters.
model = KD_SGA(sga_run=10, depth=3)

# Pass the loaded dataset through the direct adapter.
model.fit_dataset(pde_dataset)

print(f"The discovered equation is: {model.best_pde_}")
model.plot_results()
