import os

from _common import bootstrap_project_root

bootstrap_project_root()

from kd.dataset import SymbolicRegressionDataset
from kd.metrics import MSE
from kd.model.kd_llmsr import KD_LLMSR

dataset = SymbolicRegressionDataset(name="Koza-2")
data = dataset.get_data()

# Start the backbone first, for example with scripts/serve_llmsr_backbone.sh.
endpoint = os.environ.get("LLMSR_ENDPOINT", "http://127.0.0.1:10005/v1/chat/completions")

# Protocol defaults (Llama-3.1-8B-Instruct, 4 programs per prompt, 10 islands,
# BFGS over at most 10 parameters); a small call budget keeps the example short.
model = KD_LLMSR(endpoint=endpoint, max_llm_calls=8, random_state=0)
model.fit(data["X_train"], data["y_train"], variable_names=["x1"], target_name="y")

prediction = model.predict(data["X_test"])
print(f"Expression: {model.best_expression_}")
print(f"Test MSE: {MSE()(data['y_test'], prediction):.6e}")
print(f"Search statistics: {model.search_stats_}")
