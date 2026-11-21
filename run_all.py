#!/usr/bin/env python
"""Reproduce core (and optionally supplementary) experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Skip supplementary experiments (alt targets, inspection study, ablation).",
    )
    args = parser.parse_args()
    py = sys.executable

    run([py, "-m", "src.download_data"])
    run([py, "-m", "pytest", "-q", "tests"])

    # Always rebuild panel on full/core runs for consistency after code changes.
    train_cmd = [py, "-m", "src.train_eval"]
    if args.core_only:
        train_cmd.append("--core-only")
    run(train_cmd)

    example = ROOT / "examples" / "bridge_history_example.json"
    model_path = ROOT / "results" / "hgb_model.joblib"
    if model_path.exists():
        run([py, "-m", "src.predict_from_record", str(example)])

    summary = ROOT / "results" / "metrics_summary.csv"
    full = ROOT / "results" / "metrics_full.json"
    print("\n" + "=" * 72)
    print("DONE.")
    print(f"  mode: {'core-only' if args.core_only else 'core + supplementary'}")
    print(f"  - {summary}")
    print(f"  - {full}")
    print(f"  - {ROOT / 'results' / 'figures'} (core protocol figures)")
    if not args.core_only:
        for rel in (
            "results/supplementary_results.json",
            "results/inspection_study.json",
        ):
            path = ROOT / rel
            if path.exists():
                print(f"  - {path}")
        # Ablation artifact from supplementary path
        ablations = sorted((ROOT / "results").glob("ablation_*.json"))
        for path in ablations:
            print(f"  - {path}")
    if full.exists():
        payload = json.loads(full.read_text(encoding="utf-8"))
        print(f"  splits evaluated: {list(payload.get('splits', {}).keys())}")
    print("=" * 72)


if __name__ == "__main__":
    main()
