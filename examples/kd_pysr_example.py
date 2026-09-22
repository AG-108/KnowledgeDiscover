from _common import bootstrap_project_root

bootstrap_project_root()

from pysr import PySRRegressor

from kd.dataset import SymbolicRegressionDataset
from kd.metrics import MSE

dataset = SymbolicRegressionDataset(name="Koza-2")
data = dataset.get_data()

model = PySRRegressor(
    niterations=5,
    populations=1,
    population_size=30,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos"],
    parallelism="serial",
    progress=False,
    verbosity=0,
    temp_equation_file=True,
)
model.fit(data["X_train"], data["y_train"], variable_names=["x1"])

prediction = model.predict(data["X_test"])
print(f"Expression: {model.sympy()}")
print(f"Test MSE: {MSE()(data['y_test'], prediction):.6e}")
