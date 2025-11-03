from gplearn.genetic import SymbolicRegressor
from kd.dataset import SymbolicRegressionDataset
from kd.metrics import MSE

# Load a symbolic regression dataset
dataset = SymbolicRegressionDataset(name='Koza-2')

data = dataset.get_data()
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

model = SymbolicRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = MSE()

print(f"Test MSE: {mse(y_test, y_pred)}")
