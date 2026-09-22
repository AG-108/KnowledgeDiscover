"""Merge completed benchmark shard summaries into one aggregate result set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_benchmark import save_summary, write_json


def merge_results(output_dir: Path, shard_dirs: list[Path]) -> list[dict]:
    rows = []
    shards = []
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "benchmark_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing shard summary: {summary_path}")
        shard_rows = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(shard_rows, list):
            raise ValueError(f"Shard summary must contain a JSON list: {summary_path}")
        rows.extend(shard_rows)
        shards.append(
            {
                "directory": str(shard_dir.resolve()),
                "rows": len(shard_rows),
                "statuses": dict(Counter(row.get("status") for row in shard_rows)),
            }
        )

    save_summary(output_dir, rows)
    write_json(output_dir / "benchmark_shards.json", shards)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("shard_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    rows = merge_results(args.output_dir, args.shard_dirs)
    print(
        f"Merged {len(rows)} rows into {args.output_dir}: "
        f"{dict(Counter(row.get('status') for row in rows))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
