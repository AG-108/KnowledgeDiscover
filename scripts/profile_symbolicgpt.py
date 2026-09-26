"""Profile one SymbolicGPT target from a saved case, without tuning.

This timing diagnostic is not a case-budget benchmark. Use an external timeout;
flushed events retain completed-stage evidence after interruption. CUDA is
synchronized only at stage boundaries. Host loader waits overlap GPU work and
must not be added to the stage totals. Dataset, split and scoring use the runner.
"""

import argparse
import functools
import hashlib
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import kd.model.kd_symbolicgpt as adapter  # noqa: E402
import kd.model.symbolicgpt.trainer as training  # noqa: E402
import run_benchmark as runner  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    case = json.loads(args.case.read_text(encoding="utf-8"))
    if case["model"] != "symbolicgpt":
        raise ValueError("A saved SymbolicGPT case is required")
    if args.workers is not None:
        case["model_params"]["dataloader_workers"] = args.workers
    runner.write_json(output / "case.json", case)
    sources = [
        "kd/model/kd_symbolicgpt.py", "kd/model/symbolicgpt/trainer.py",
        "kd/model/symbolicgpt/models.py", "kd/model/symbolicgpt/utils.py",
        "kd/model/symbolicgpt/generator.py",
        "run_benchmark.py", "scripts/profile_symbolicgpt.py",
    ]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes().replace(
        b"\r\n", b"\n")).hexdigest() for name in sources}
    runner.write_json(output / "source_hashes.json", hashes)
    device = torch.device(case["device"])
    started = time.perf_counter()
    totals, counts, evidence = {}, {}, {}
    epoch_count = 0
    loader_count = 0

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with (output / "events.jsonl").open("x", encoding="utf-8", buffering=1) as events:
        def emit(event, **values):
            events.write(json.dumps(runner._json_safe(dict(
                event=event, elapsed=time.perf_counter() - started, **values,
            )), allow_nan=False) + "\n")

        def timed(name, function, synchronize=False):
            @functools.wraps(function)
            def wrapped(*pos, **kw):
                if synchronize:
                    sync()
                begin = time.perf_counter()
                emit("stage_start", stage=name)
                status = "ok"
                try:
                    return function(*pos, **kw)
                except Exception:
                    status = "error"
                    raise
                finally:
                    if synchronize:
                        sync()
                    seconds = time.perf_counter() - begin
                    totals[name] = totals.get(name, 0.0) + seconds
                    counts[name] = counts.get(name, 0) + 1
                    emit("stage_end", stage=name, seconds=seconds, status=status)
            return wrapped

        class ObservedDataLoader(training.DataLoader):
            def __init__(self, *pos, **kw):
                nonlocal loader_count
                super().__init__(*pos, **kw)
                loader_count += 1

            def __iter__(self):
                nonlocal epoch_count
                epoch_count += 1
                epoch = epoch_count
                begin = time.perf_counter()
                iterator = super().__iter__()
                waiting = time.perf_counter() - begin
                batches = 0
                while True:
                    fetch = time.perf_counter()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        waiting += time.perf_counter() - fetch
                        break
                    waiting += time.perf_counter() - fetch
                    batches += 1
                    yield batch
                emit("epoch_end", epoch=epoch, batches=batches,
                     seconds=time.perf_counter() - begin, host_loader_wait=waiting)

        original_train = adapter.Trainer.train
        original_fit = adapter.KD_SymbolicGPT.fit

        def train(self):
            timed("pretrain", original_train, synchronize=True)(self)
            digest = hashlib.sha256()
            for key, value in self.model.state_dict().items():
                digest.update(key.encode())
                digest.update(value.detach().cpu().numpy().tobytes())
            evidence["trained_state_sha256"] = digest.hexdigest()
            emit("trained_state", **evidence, loader_count=loader_count)

        def fit(self, X, y, **kwargs):
            try:
                return timed("fit_total", original_fit, synchronize=True)(self, X, y, **kwargs)
            finally:
                evidence["candidate_validation"] = self.candidate_validation_
                evidence["candidates"] = self.candidates_
                emit("candidate_validation", **evidence["candidate_validation"])

        instance, dataset = next(iter(runner._load_instances(case)))
        problems = list(runner.regression_problems(dataset, case))
        problem = problems[args.target_index]
        emit("profile_start", instance=instance, target=problem["target"],
             target_count=len(problems), n_train=len(problem["X_train"]),
             params=case["model_params"], torch_version=torch.__version__)
        result = {}
        with ExitStack() as stack:
            stack.enter_context(patch.object(training, "DataLoader", ObservedDataLoader))
            stack.enter_context(patch.object(adapter.Trainer, "train", train))
            stack.enter_context(patch.object(adapter.KD_SymbolicGPT, "fit", fit))
            stack.enter_context(patch.object(adapter.KD_SymbolicGPT, "_build_pretrain_corpus",
                timed("corpus", adapter.KD_SymbolicGPT._build_pretrain_corpus)))
            for name, synchronize in (("sample_from_model", True), ("fit_constants", False)):
                stack.enter_context(patch.object(adapter, name,
                    timed(name, getattr(adapter, name), synchronize=synchronize)))
            try:
                result = runner.run_regression(problem, case)
                result["status"] = "ok"
            except Exception as exc:
                result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            finally:
                result.update(diagnostic="single_target_profile_not_case_benchmark",
                              target=problem["target"], timings=totals, calls=counts,
                              loader_count=loader_count, **evidence)
                runner.write_json(output / "profile_result.json", result)
                emit("profile_end", status=result["status"], timings=totals)
        print(json.dumps(dict(status=result["status"], timings=totals,
                              candidate_validation=evidence.get("candidate_validation"))))


if __name__ == "__main__":
    main()
