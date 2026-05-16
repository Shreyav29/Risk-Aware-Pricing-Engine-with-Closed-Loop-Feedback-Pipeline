"""Run the risk-aware pricing project end to end.

This file is the repository entry point. The numbered scripts are kept separate
so each stage can also be inspected or rerun independently.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PIPELINE_STEPS = [
    ("1_king_county_eda.py", "Data and market environment study"),
    ("2_point_avm_baseline.py", "Point-model baseline comparison"),
    ("3_risk_adjusted_pricing.py", "Main quantile AVM and risk-adjusted offer engine"),
    ("4_quantile_calibration.py", "Width-based quantile calibration analysis"),
    ("5_calibration_method_comparison.py", "Width vs isotonic calibration comparison"),
    ("6_sell_side_feedback_loop.py", "Scenario-based feedback loop analysis"),
    ("7_realtime_feedback_backtest.py", "Rolling real-time feedback backtest"),
    ("8_downside_stress_feedback_backtest.py", "Downside stress feedback analysis"),
]


def run_step(script_name: str, description: str) -> None:
    """Run one project stage and stop immediately if it fails."""

    script_path = Path(script_name)
    if not script_path.exists():
        raise FileNotFoundError(f"Missing pipeline script: {script_path}")

    print(f"\n=== {script_name}: {description} ===")
    subprocess.run([sys.executable, str(script_path)], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full risk-aware home pricing pipeline.",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=range(1, len(PIPELINE_STEPS) + 1),
        metavar=f"1-{len(PIPELINE_STEPS)}",
        help="First numbered pipeline step to run.",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        default=len(PIPELINE_STEPS),
        choices=range(1, len(PIPELINE_STEPS) + 1),
        metavar=f"1-{len(PIPELINE_STEPS)}",
        help="Last numbered pipeline step to run.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List pipeline steps without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.from_step > args.to_step:
        raise ValueError("--from-step must be less than or equal to --to-step")

    if args.list:
        for index, (script_name, description) in enumerate(PIPELINE_STEPS, start=1):
            print(f"{index}. {script_name} - {description}")
        return

    selected_steps = PIPELINE_STEPS[args.from_step - 1 : args.to_step]
    for script_name, description in selected_steps:
        run_step(script_name, description)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
