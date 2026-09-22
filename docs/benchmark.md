# 统一 benchmark

指标定义与聚合口径见 [Benchmark 指标协议](benchmark_metrics.md)。

`run_benchmark.py` 统一组织 symbolic regression（`sr`）、ODE discovery（`ode`）和 PDE discovery（`pde`）任务。数据清单来自 `kd.dataset.DATASET_REGISTRY`、`kd/dataset/data/benchmarks.csv` 和 WaveBreaking 适配器；模型按需导入，每个 model/dataset/seed 组合在独立子进程和工作目录中运行。

## 领域数据集

以下三组数据已通过统一的 `load_dataset(name)` 接口进入 catalog；总配置中的
`"datasets": ["*"]` 会自动包含全部 9 个入口。

| 数据来源 | Benchmark 名称 | 任务 | 默认输入与目标 |
| --- | --- | --- | --- |
| CYT | `cyt_flatplate`、`cyt_naca0012` | SR | 15 个局部流场特征预测涡黏度 `Mut` |
| Discovery of solid constitutive laws | `solid_dif` | SR | 应变率预测动态增长因子 DIF |
| Discovery of solid constitutive laws | `solid_strain_stress` | SR | 塑性应变和应变率预测塑性应力 |
| Discovery of solid constitutive laws | `solid_hardening` | SR | 应变、应变率和 DIF 预测塑性应力 |
| VGS | `vgs_I_0-100`、`vgs_I_100-200`、`vgs_II_0-1000`、`vgs_II_1000-2000` | PDE | 500 × 500 自由界面高度场 `h(x,t)` |

固体本构数据使用源文件或材料曲线作为分组，benchmark 按组切分训练集与测试集，避免同一条
实验曲线同时出现在两边。可以只检查或运行这三组数据：

```bash
python -u run_benchmark.py --datasets "cyt_*" "solid_*" "vgs_*" --dry-run
python -u run_benchmark.py --datasets "cyt_*" "solid_*" "vgs_*" \
  --check-compatibility --output-dir results/domain_datasets
```

## 配置结构

配置不再硬编码在运行脚本中：

```text
configs/benchmark/
├── benchmark.json          # 总配置：任务选择、profile、设备、输出和通用运行参数
└── baselines/
    ├── deepmod.json        # 每个 baseline 的模型参数及 smoke/full 差异
    ├── dlga.json
    ├── dscv.json
    └── ...                 # 共 18 个 baseline 配置
```

总配置 [benchmark.json](../configs/benchmark/benchmark.json) 的主要字段如下：

| 字段 | 作用 |
| --- | --- |
| `profile` | 选择 `smoke` 或 `full` 预算 |
| `tasks` | 选择 `sr`、`ode`、`pde` |
| `models` / `datasets` | 名称或通配符列表；`["*"]` 表示全选 |
| `seeds` | 随机种子列表 |
| `devices` | 每个 baseline 的 `cpu`、`cuda` 或 `cuda:N` |
| `output_dir` / `timeout` / `resume` | 输出目录、单个 case 超时秒数和断点恢复 |
| `profiles.*.run` | benchmark 适配层的通用预算 |
| `overrides` | 对具体模型、数据集或 run 参数做最终覆盖 |

每个 baseline 文件包含基础 `model` 参数、默认 `device`，以及 `profiles.smoke/full` 下的增量参数。自定义 `--config` 会覆盖仓库总配置，因此服务器任务通常只需写少量字段：

```json
{
  "profile": "full",
  "tasks": ["sr", "ode"],
  "models": ["dso", "symbolicgpt", "physo"],
  "datasets": ["Koza-*", "Keijzer-*", "ball_drop"],
  "seeds": [0, 1, 2],
  "output_dir": "results/server_sr",
  "timeout": 21600,
  "resume": true,
  "devices": {
    "dso": "cuda:0",
    "symbolicgpt": "cuda:1",
    "physo": "cuda:1"
  },
  "overrides": {
    "models": {
      "dso": {"n_samples": 500000}
    },
    "datasets": {},
    "run": {"max_train_samples": 10000}
  }
}
```

命令行选择和 `--device` 高于总配置；总配置的 `overrides` 高于对应 baseline profile。最终解析出的 `device` 和全部参数会写入 manifest、`case.json` 和结果行。

## GPU 支持

| Baseline | 当前 GPU 支持 | GPU 上运行的主要部分 |
| --- | --- | --- |
| `dso` | 支持指定设备 | PyTorch 策略网络；表达式奖励仍含 CPU/NumPy 计算 |
| `symbolicgpt` | 支持指定设备 | Transformer 训练和生成；候选表达式计算仍含 CPU 工作 |
| `eqgpt` | 支持指定设备 | GPT、代理网络和自动微分 |
| `pdenet` | 支持指定设备 | PDE-Net 模型训练和预测 |
| `deepmod` | 支持指定设备 | 神经网络、自动微分和稀疏约束 |
| `dlga` | 支持指定设备 | 神经网络和导数计算；遗传搜索在 CPU |
| `physo` | 支持指定设备 | `physo.SR(..., device=...)` 的张量计算 |
| `spr` | 部分支持 | PINN 训练和自动微分；DISCOVER 搜索部分仍在 CPU |
| `dscv` | 当前仅 CPU | 当前 controller/state 代码没有完整设备迁移 |
| `sga` | 当前仅 CPU | 当前集成没有稳定的显式设备接口 |
| `pdefind` | CPU | NumPy/SciPy 回归 |
| `weakform` | CPU | NumPy 有限差分和最小二乘 |
| `gplearn` | CPU | scikit-learn/joblib 风格的 CPU 遗传编程 |
| `pysr` | CPU | Julia 多线程/多进程符号搜索；当前 PySR 后端不使用 CUDA |
| `sindy` | CPU | 仓库内 NumPy 多项式库与 STLSQ 稀疏回归 |
| `pysindy` | CPU | PySINDy 多项式库与 SR3 稀疏回归 |
| `llmsr` | CPU + 外部 LLM 服务 | 常数优化和候选评分在 CPU；本地 LLM 服务可自行配置 GPU |
| `pyoperon` | CPU | Operon C++ 遗传编程和多线程候选评分 |

总配置默认全部使用 CPU，保证没有 GPU 的环境也能生成和运行任务。迁移到服务器后修改 `devices` 即可。普通执行会在启动 case 前检查 CUDA 是否可用及编号是否越界；也可以只做检查：

```bash
python -u run_benchmark.py --config configs/benchmark/benchmark.json \
  --models dso eqgpt --device dso=cuda:0 --device eqgpt=cuda:1 \
  --check-devices
```

`run_benchmark.py` 当前按 case 串行调度。设备映射保证每个 baseline 使用指定显卡，但不会自动让多个 case 并发。需要同时使用多张卡时，可以启动多个进程并给它们不同的模型集合和输出目录。

## SSH 运行

先检查任务矩阵和设备，再用仓库提供的启动脚本运行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kd-env

# PySR 需要 Julia 1.10 或更高版本；首次导入会安装 Julia 侧依赖。
juliaup add 1.10
juliaup default 1.10
python -c "from pysr import PySRRegressor; print('PySR ready')"

# PyOperon 0.6.1 需要 Python >=3.10；新服务器环境应使用 3.10 或 3.11。
pip install -r requirements-server-py310.txt
python -c "from pyoperon.sklearn import SymbolicRegressor; print('PyOperon ready')"

python -u run_benchmark.py --config configs/benchmark/benchmark.json --dry-run
python -u run_benchmark.py --config configs/benchmark/benchmark.json --check-devices

mkdir -p logs
nohup bash scripts/run_benchmark.sh configs/benchmark/benchmark.json \
  > logs/benchmark.log 2>&1 &
echo $!
tail -f logs/benchmark.log
```

仓库当前开发环境是 Windows/Python 3.9，PyOperon 0.6.1 没有适用的官方 wheel，因此该
baseline 在本机兼容性预检中会显示为依赖错误。Linux/Python 3.9 可以使用兼容版
PyOperon 0.5.0，`requirements.txt` 已通过平台标记处理；新服务器仍推荐 Python 3.10 或
3.11 与 PyOperon 0.6.1。其余 baseline 配置可在缺少 PyOperon 时正常读取，对应 case 会独立
标记为 `skipped`。

命令行可以临时覆盖总配置，无需改 JSON：

```bash
bash scripts/run_benchmark.sh configs/benchmark/benchmark.json \
  --profile full --models dso --datasets Koza-2 --seeds 0 1 2 \
  --device dso=cuda:0 --resume
```

两个 GPU 并行运行时必须使用不同输出目录，避免两个进程同时改写同一份 summary：

```bash
nohup bash scripts/run_benchmark.sh configs/benchmark/benchmark.json \
  --models dso symbolicgpt --device dso=cuda:0 --device symbolicgpt=cuda:0 \
  --output-dir results/gpu0 --resume > logs/gpu0.log 2>&1 &

nohup bash scripts/run_benchmark.sh configs/benchmark/benchmark.json \
  --models eqgpt deepmod --device eqgpt=cuda:1 --device deepmod=cuda:1 \
  --output-dir results/gpu1 --resume > logs/gpu1.log 2>&1 &
```

## 常用命令

```bash
python -u run_benchmark.py --list
python -u run_benchmark.py --dry-run
python -u run_benchmark.py --tasks sr --models gplearn dso --datasets Koza-2 Keijzer-2
python -u run_benchmark.py --tasks ode --models physo --datasets ball_drop
python -u run_benchmark.py --tasks ode --models sindy pysindy --datasets ball_drop
python -u run_benchmark.py --tasks sr --models pysr --datasets Koza-2
python -u run_benchmark.py --tasks sr --models pyoperon --datasets Koza-2
python -u run_benchmark.py --tasks pde --models pdefind deepmod --datasets kdv burgers
python -u run_benchmark.py --profile full --seeds 0 1 2 --resume
```

## LLM-SR 服务配置

`llmsr` 使用官方 LLM-SR 的程序骨架、多岛候选搜索和数值常数拟合思路，但通过统一
`fit(X, y)` 适配器读取本 benchmark 的 SR 与 ODE 数据。默认不会把 API 密钥写入 JSON；
密钥只在发送请求时从环境变量读取。

使用官方本地 completion server 或同协议服务：

```bash
export LLMSR_ENDPOINT=http://127.0.0.1:5000/completions
python -u run_benchmark.py --models llmsr --datasets Koza-2 --timeout 3600
```

使用 OpenAI 兼容的 chat-completions API 时，可建立一个只覆盖必要参数的配置文件：

```json
{
  "models": ["llmsr"],
  "datasets": ["Koza-2"],
  "overrides": {
    "models": {
      "llmsr": {
        "use_api": true,
        "api_model": "your-model-name"
      }
    },
    "datasets": {},
    "run": {}
  }
}
```

```bash
export LLMSR_API_KEY='...'
export LLMSR_ENDPOINT=https://your-host.example/v1/chat/completions
python -u run_benchmark.py --config configs/llmsr-server.json --resume
```

也可用 `LLMSR_MODEL` 提供模型名。`endpoint`、搜索预算和超时均可在
`configs/benchmark/baselines/llmsr.json` 或总配置的 model override 中修改。本项目只调度
LLM-SR 的候选搜索；LLM 推理服务使用哪张 GPU 由服务启动命令控制。

`--models` 和 `--datasets` 支持名称与带引号的通配符。`smoke` 和 `full` 使用同一份数据覆盖范围，仅训练预算不同。`--timeout` 限制每个 worker 的总秒数；`--resume` 只复用配置身份相同且全部成功的 case。

## 输出

```text
results/benchmark/
├── benchmark_manifest.json
├── benchmark_aggregate.csv
├── benchmark_aggregate.json
├── benchmark_summary.csv
├── benchmark_summary.json
└── cases/<case-name>/
    ├── case.json
    ├── run.log
    ├── result.json
    └── partial.json
```

状态包括 `ok`、`skipped`、`error`、`timeout`。单个 case 失败不会中止后续实验；最终存在 `error` 或 `timeout` 时主进程返回非零退出码。PDE 的可用网格、维度和缺失值限制仍由各适配器在运行时检查，不兼容组合会记录具体的 `skipped` 原因。

验证 benchmark 结构和适配器：

```bash
python -m pytest tests/test_run_benchmark.py -q
```

## 兼容性检查

以下命令会为每个选中的 baseline/dataset 组合加载一个最小实例并执行适配器预检，不进行
模型优化：

```bash
python -u run_benchmark.py --check-compatibility \
  --output-dir results/compatibility_audit
```

结果写入 `compatibility_report.json` 和 `compatibility_pairs.csv`。完整检查结论及 PDE 矩阵见
[Baseline 与 Dataset 兼容性报告](compatibility_report.md)。
