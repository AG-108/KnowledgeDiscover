"""Create an unlabeled physical-credibility challenge/annotation template offline."""

import argparse
import csv
import json
from pathlib import Path


MUTATIONS = (
    ("equivalent_rewrite", "v = x/t", "t*v = x"),
    ("wrong_sign", "du_dt = -k*u", "du_dt = k*u"),
    ("missing_term", "du_dt = diffusion + source", "du_dt = diffusion"),
    ("redundant_term", "du_dt = diffusion", "du_dt = diffusion + 0*u"),
    ("invalid_domain", "y = log(x)", "y = log(-x)"),
    ("ood_error", "y = x/(1+x)", "y = x/(1-x)"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, (kind, left, right) in enumerate(MUTATIONS, 1):
        cases.append({"challenge_id": "pc-%03d" % index, "mutation": kind,
                      "candidate_a": left, "candidate_b": right,
                      "generator_intent_only": True, "expert_validated": False})
    (args.output_dir / "challenges.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    with (args.output_dir / "expert_annotations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("challenge_id", "presentation_order", "preferred", "confidence",
                         "violation_types", "rationale", "annotator_id", "timestamp"))
        for case in cases:
            writer.writerow((case["challenge_id"], "", "", "", "", "", "", ""))


if __name__ == "__main__":
    main()
