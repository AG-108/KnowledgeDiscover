import os

from _common import bootstrap_project_root

bootstrap_project_root()

from kd.dataset import SymbolicRegressionDataset
from kd.metrics import MSE
from kd.model.kd_llmsr import KD_LLMSR

dataset = SymbolicRegressionDataset(name="Koza-2")
data = dataset.get_data()

if not os.environ.get("LLMSR_ENDPOINT"):
    raise RuntimeError("Start an LLM-SR-compatible completion server and set LLMSR_ENDPOINT first")

model = KD_LLMSR(max_samples=8, samples_per_prompt=2, random_state=0)
model.fit(data["X_train"], data["y_train"], variable_names=["x1"])

prediction = model.predict(data["X_test"])
print(f"Expression: {model.best_expression_}")
print(f"Test MSE: {MSE()(data['y_test'], prediction):.6e}")
print(f"Search statistics: {model.search_stats_}")
