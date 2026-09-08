import os
import sys

from _common import bootstrap_project_root

project_root = bootstrap_project_root()

from kd.model.kd_sga import KD_SGA

# 1. 创建并配置参数 (所有在 SGA config 中定义的参数都已被兼容)
model = KD_SGA(sga_run=10, depth=3) 

# 2. 加入数据（当前通过 problem_name）并训练
model.fit(problem_name='chafee-infante')

# 3. 查看结果
print(f"The discovered equation is: {model.best_pde_}")

# 4. 可视化
model.plot_results()


