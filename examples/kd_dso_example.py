import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

from dso import DeepSymbolicRegressor
import numpy as np

config_dir = os.path.join(kd_main_dir, "kd/model/DeepSymbolicOptimization/dso/dso/config")
config_path = os.path.join(config_dir, "config_regression.json")

# Create the model
model = DeepSymbolicRegressor(config_path) # Alternatively, you can pass in your own config JSON path

# Fit the model
model.fit_benchmark("Keijzer-2") # Should solve in ~10 seconds

# View the best expression
print(model.program_.pretty())
