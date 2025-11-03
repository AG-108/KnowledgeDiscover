import os.path

import numpy as np
import matplotlib.pyplot as plt
import torch

import physo
import physo.learn.monitoring as monitoring
from sympy.unify.usympy import illegal

from kd.dataset import SymbolicRegressionDataset

def filter_out_illegal_ops(function_set):
    """Filter out illegal operations from the function set."""
    illegal_ops = ("const")
    legal_ops = [op for op in function_set if op not in illegal_ops]
    legal_ops = [op for op in legal_ops if not isinstance(op, float)]
    return legal_ops

# Seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

dataset = SymbolicRegressionDataset(name='Keijzer-2')
# function_set中有一些非法运算符；后面需要处理
data = dataset.get_data()

X_train = data['X_train'].T
y_train = data['y_train']
X_test = data['X_test'].T
y_test = data['y_test']
function_set = data['function_set']

function_set = filter_out_illegal_ops(function_set)

save_path = './physo_res'

if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path_training_curves = os.path.join(save_path, 'demo_curves.png')
save_path_log             = os.path.join(save_path, 'demo.log')

run_logger     = lambda : monitoring.RunLogger(save_path = save_path_log,
                                                do_save = True)

run_visualiser = lambda : monitoring.RunVisualiser (epoch_refresh_rate = 1,
                                           save_path = save_path_training_curves,
                                           do_show   = False,
                                           do_prints = True,
                                           do_save   = True, )

expression, logs = physo.SR(X_train, y_train,
                            # Giving names of variables (for display purposes)
                            X_names = [ "x1" ],
                            # Giving name of root variable (for display purposes)
                            y_name  = "y",
                            y_units = [0, 0, 0],
                            # Fixed constants
                            fixed_consts       = [ 1.      ],
                            fixed_consts_units = [ [0,0,0] ],
                            # Free constants names (for display purposes)
                            # Symbolic operations that can be used to make f
                            op_names = function_set,
                            get_run_logger     = run_logger,
                            get_run_visualiser = run_visualiser,
                            # Run config
                            run_config = physo.config.config0.config0,
                            # Parallel mode (only available when running from python scripts, not notebooks)
                            parallel_mode = False,
                            # Number of iterations
                            epochs = 20
)

best_expr = expression
print(best_expr.get_infix_pretty())