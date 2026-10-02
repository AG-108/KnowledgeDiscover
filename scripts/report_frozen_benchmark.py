"""Generate tables and publication figures using only a result directory and manifest SHA256."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kd.evaluation.frozen_report import generate  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("manifest_sha256")
    args = parser.parse_args(argv)
    try:
        output = generate(args.result_dir, args.manifest_sha256)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
