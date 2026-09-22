"""Minimal example: SymbolicGPT baseline on the Koza-2 symbolic regression
benchmark (same dataset used by kd_gplearn_example.py / kd_physo_example.py,
for easy side-by-side comparison).

SymbolicGPT pretrains a small character-level GPT from scratch on a
synthetic corpus of random equations (generated on the fly for the
dataset's own number of variables), then autoregressively samples candidate
equation skeletons conditioned on the real data via a PointNet-style
point-cloud encoder, and fits each candidate's constants against the real
data (see kd/model/symbolicgpt/ and kd/model/kd_symbolicgpt.py for the
adapted implementation; the original repo is vendored at
kd/dataset/SymbolicGPT/).

Settings below are intentionally small (small transformer, small corpus,
few epochs) to keep this example fast (~1-2 minutes on CPU) -- this trades
off reliability for speed, so the final answer is not guaranteed to match
the textbook Koza-2 form (x^5 - 2x^3 + x).
"""

from _common import bootstrap_project_root

project_root = bootstrap_project_root()

from kd.dataset import SymbolicRegressionDataset
from kd.model.kd_symbolicgpt import KD_SymbolicGPT

# Load the same benchmark used by the gplearn and PhySO examples.
dataset = SymbolicRegressionDataset(name="Koza-2")

# Use a small SymbolicGPT configuration for a quick demonstration.
model = KD_SymbolicGPT(
    embedding_size=32,
    n_layer=2,
    n_head=2,
    pretrain_corpus_size=300,
    pretrain_epochs=15,
    batch_size=32,
    op_list=["add", "sub", "mul", "div", "sin", "cos", "pow"],
    num_candidates=20,
    seed=0,
    device="cpu",
    verbose=True,
)
model.fit_dataset(dataset)

print(f"\nBest skeleton: {model.best_skeleton_}")
print(f"Best expression: {model.best_expression_}")
print(f"Train loss: {model.train_loss_:.6g}")
if hasattr(model, "test_mse_"):
    print(f"Test MSE: {model.test_mse_:.6g}")
