#!/usr/bin/env python
"""Reproduce the full prototype end-to-end with one command.

Steps:
  1) download FHWA NBI delimited files (or print manual instructions)
  2) run unit tests (target construction + time-split leakage checks)
  3) build panel, train baselines + HGB, write metrics/figures
  4) smoke-test predict_from_record on the example JSON
"""

from __future__ import annotations

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
    py = sys.executable

    # 1. Data
    run([py, "-m", "src.download_data"])

    # 2. Tests (must pass before claiming reproducibility)
    run([py, "-m", "pytest", "-q", "tests"])

    # 3. Train / evaluate
    run([py, "-m", "src.train_eval"])

    # 4. Integration smoke test
    example = ROOT / "examples" / "bridge_history_example.json"
    run([py, "-m", "src.predict_from_record", str(example)])

    summary = ROOT / "results" / "metrics_summary.csv"
    full = ROOT / "results" / "metrics_full.json"
    print("\n" + "=" * 72)
    print("DONE. Key artifacts:")
    print(f"  - {summary}")
    print(f"  - {full}")
    print(f"  - {ROOT / 'results' / 'figures'}")
    print(f"  - {ROOT / 'docs' / 'field_mapping.md'}")
    if full.exists():
        payload = json.loads(full.read_text(encoding="utf-8"))
        meta = payload.get("panel_meta", {})
        tr = meta.get("target_report", {})
        print("\nData / target snapshot:")
        print(
            f"  state years={meta.get('years')}  "
            f"bridges={meta.get('n_bridges')}  rows={meta.get('n_rows')}"
        )
        print(
            f"  primary_defined={tr.get('n_primary_defined')}  "
            f"positives={tr.get('n_primary_positive')}  "
            f"excluded_current_N={tr.get('n_excluded_current_N_or_missing')}  "
            f"excluded_next_N={tr.get('n_excluded_next_N_or_missing')}"
        )
        print("\nHonest comparison (HGB vs Markov) per split:")
        for name, split in payload.get("splits", {}).items():
            cmp_ = split.get("comparison", {})
            print(
                f"  {name}: HGB PR-AUC={cmp_.get('hgb_pr_auc_mean'):.4f} "
                f"vs Markov PR-AUC={cmp_.get('markov_pr_auc'):.4f} "
                f"| beats_markov_pr={cmp_.get('hgb_beats_markov_pr_auc')} "
                f"| beats_markov_roc={cmp_.get('hgb_beats_markov_roc_auc')}"
            )
    print("=" * 72)


if __name__ == "__main__":
    main()
