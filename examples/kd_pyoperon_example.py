from _common import bootstrap_project_root

bootstrap_project_root()

from pyoperon.sklearn import SymbolicRegressor

from kd.dataset import SymbolicRegressionDataset
from kd.metrics import MSE

dataset = SymbolicRegressionDataset(name="Koza-2")
data = dataset.get_data()
variable_names = ["x1"]

model = SymbolicRegressor(
    allowed_symbols="add,sub,mul,div,sin,cos,constant,variable",
    population_size=100,
    generations=10,
    max_evaluations=50000,
    n_threads=1,
    random_state=0,
)
model.fit(data["X_train"], data["y_train"])

prediction = model.predict(data["X_test"])
expression = model.get_model_string(model.model_, precision=8, names=variable_names)
print(f"Expression: {expression}")
print(f"Test MSE: {MSE()(data['y_test'], prediction):.6e}")
