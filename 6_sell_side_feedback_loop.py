"""Sell-side feedback-loop stress test for the risk-aware AVM project.

This script starts from the calibrated buy-side pricing engine and simulates
what happens after homes are resold under different market scenarios. The same
feedback rule is used for every stress test: compare observed sale outcomes to
expected Q50, estimate a market multiplier, and apply it to future offers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs/6_feedback_loop")
MAIN_PRICING_SCRIPT = Path("3_risk_adjusted_pricing.py")

SHOCK_SCENARIOS = {
    "downside_10pct": 0.90,
    "downside_5pct": 0.95,
    "neutral": 1.00,
    "upside_5pct": 1.05,
}


def load_pricing_module():
    """Load the numbered pricing script as a normal Python module."""

    spec = importlib.util.spec_from_file_location("risk_pricing", MAIN_PRICING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MAIN_PRICING_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_calibrated_buy_side_state(pricing_module) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recreate the calibrated Q10/Q50/Q90 predictions and original offers."""

    train_features, test_features, train_prices, test_prices = pricing_module.load_modeling_data()
    _, raw_predictions, best_model_params, _ = pricing_module.train_probabilistic_avm(
        train_features,
        train_prices,
        test_features,
    )

    validation_prices, validation_predictions = pricing_module.estimate_validation_predictions_for_calibration(
        train_features,
        train_prices,
        best_model_params,
    )
    calibration_factor, _ = pricing_module.choose_width_calibration_factor(
        validation_prices,
        validation_predictions,
    )
    calibrated_predictions = pricing_module.widen_q10_q90_interval(raw_predictions, calibration_factor)

    spread_assumptions = pricing_module.SpreadAssumptions()
    original_offers = pricing_module.calculate_buy_offer(
        median_price=calibrated_predictions["q50"],
        lower_price=calibrated_predictions["q10"],
        upper_price=calibrated_predictions["q90"],
        spread_assumptions=spread_assumptions,
    )
    test_context = pricing_module.load_test_sale_context()

    return test_prices, calibrated_predictions, original_offers, test_context


def apply_market_multiplier(
    predictions: pd.DataFrame,
    multiplier: float,
    pricing_module,
) -> pd.DataFrame:
    """Apply one market-level correction to all predicted quantiles."""

    adjusted_predictions = predictions.copy()
    for column in ["q10", "q50", "q90"]:
        adjusted_predictions[column] = predictions[column] * multiplier

    # Why: the market multiplier corrects level bias. It should not change the
    # relative ordering of Q10/Q50/Q90 for a property.
    adjusted_predictions[["q10", "q50", "q90"]] = np.sort(
        adjusted_predictions[["q10", "q50", "q90"]].to_numpy(),
        axis=1,
    )
    return adjusted_predictions


def run_feedback_scenarios(
    actual_prices: pd.Series,
    calibrated_predictions: pd.DataFrame,
    original_offers: pd.DataFrame,
    test_context: pd.DataFrame,
    pricing_module,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run each stress scenario through the same sell-side feedback rule."""

    scenario_rows = []
    detail_frames = []
    monthly_rows = []
    spread_assumptions = pricing_module.SpreadAssumptions()

    for scenario_name, shock_factor in SHOCK_SCENARIOS.items():
        simulated_sale_price = actual_prices * shock_factor
        sale_ratio = simulated_sale_price / calibrated_predictions["q50"]
        market_multiplier = sale_ratio.median()

        adjusted_predictions = apply_market_multiplier(
            calibrated_predictions,
            market_multiplier,
            pricing_module,
        )
        adjusted_offers = pricing_module.calculate_buy_offer(
            median_price=adjusted_predictions["q50"],
            lower_price=adjusted_predictions["q10"],
            upper_price=adjusted_predictions["q90"],
            spread_assumptions=spread_assumptions,
        )

        offer_change = adjusted_offers["buy_offer"] - original_offers["buy_offer"]
        offer_change_pct = offer_change / original_offers["buy_offer"]

        scenario_rows.append(
            {
                "scenario": scenario_name,
                "shock_factor": shock_factor,
                "market_multiplier": market_multiplier,
                "implied_market_bias": market_multiplier - 1,
                "median_sale_ratio": sale_ratio.median(),
                "mean_sale_ratio": sale_ratio.mean(),
                "average_original_offer": original_offers["buy_offer"].mean(),
                "average_feedback_adjusted_offer": adjusted_offers["buy_offer"].mean(),
                "average_offer_change": offer_change.mean(),
                "average_offer_change_pct": offer_change_pct.mean(),
                "median_offer_change": offer_change.median(),
                "median_offer_change_pct": offer_change_pct.median(),
            }
        )

        detail_frame = pd.DataFrame(
            {
                "scenario": scenario_name,
                "actual_test_price": actual_prices,
                "simulated_sale_price": simulated_sale_price,
                "expected_sale_value_q50": calibrated_predictions["q50"],
                "sale_ratio": sale_ratio,
                "market_multiplier": market_multiplier,
                "original_buy_offer": original_offers["buy_offer"],
                "feedback_adjusted_buy_offer": adjusted_offers["buy_offer"],
                "offer_change": offer_change,
                "offer_change_pct": offer_change_pct,
                "sale_month": test_context["sale_date"].dt.to_period("M").astype(str),
            },
            index=actual_prices.index,
        )
        detail_frames.append(detail_frame)

        monthly_frame = detail_frame.groupby("sale_month").agg(
            monthly_home_count=("sale_ratio", "size"),
            monthly_market_multiplier=("sale_ratio", "median"),
            monthly_average_offer_change_pct=("offer_change_pct", "mean"),
        )
        monthly_frame["scenario"] = scenario_name
        monthly_rows.append(monthly_frame.reset_index())

    scenario_summary = pd.DataFrame(scenario_rows)
    scenario_details = pd.concat(detail_frames).reset_index(names="row_id")
    monthly_summary = pd.concat(monthly_rows).reset_index(drop=True)

    return scenario_summary, scenario_details, monthly_summary


def plot_market_multipliers(scenario_summary: pd.DataFrame) -> None:
    """Plot how each stress scenario changes the market multiplier."""

    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    colors = ["#E45756", "#F58518", "#4C78A8", "#54A24B"]
    axis.bar(scenario_summary["scenario"], scenario_summary["market_multiplier"], color=colors)
    axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_title("Sell-Side Feedback Multiplier By Stress Scenario")
    axis.set_xlabel("Stress Scenario")
    axis.set_ylabel("Median Simulated Sale / Expected Q50")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)

    for index, value in enumerate(scenario_summary["market_multiplier"]):
        axis.text(index, value + 0.01, f"{value:.3f}", ha="center", fontsize=10)

    figure.savefig(OUTPUT_DIR / "market_multiplier_by_scenario.png", dpi=160)
    plt.close(figure)


def plot_offer_changes(scenario_summary: pd.DataFrame) -> None:
    """Plot how the feedback multiplier changes future acquisition offers."""

    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    colors = ["#E45756" if value < 0 else "#54A24B" for value in scenario_summary["average_offer_change_pct"]]
    axis.bar(scenario_summary["scenario"], scenario_summary["average_offer_change_pct"], color=colors)
    axis.axhline(0, color="#333333", linestyle="--", linewidth=1)
    axis.set_title("Average Future Offer Change After Feedback")
    axis.set_xlabel("Stress Scenario")
    axis.set_ylabel("Average Offer Change")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)

    for index, value in enumerate(scenario_summary["average_offer_change_pct"]):
        axis.text(index, value + (0.003 if value >= 0 else -0.009), f"{value:.1%}", ha="center", fontsize=10)

    figure.savefig(OUTPUT_DIR / "offer_change_by_scenario.png", dpi=160)
    plt.close(figure)


def plot_monthly_multipliers(monthly_summary: pd.DataFrame) -> None:
    """Plot monthly feedback multipliers for each stress scenario."""

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for scenario_name, scenario_frame in monthly_summary.groupby("scenario"):
        axis.plot(
            scenario_frame["sale_month"],
            scenario_frame["monthly_market_multiplier"],
            marker="o",
            linewidth=2,
            label=scenario_name,
        )

    axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_title("Monthly Sell-Side Feedback Multiplier")
    axis.set_xlabel("Sale Month")
    axis.set_ylabel("Median Simulated Sale / Expected Q50")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)

    figure.savefig(OUTPUT_DIR / "monthly_market_multiplier_by_scenario.png", dpi=160)
    plt.close(figure)


def plot_sale_ratio_distribution(scenario_details: pd.DataFrame) -> None:
    """Show the sale-ratio distribution that drives each feedback multiplier."""

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for scenario_name, scenario_frame in scenario_details.groupby("scenario"):
        axis.hist(
            scenario_frame["sale_ratio"],
            bins=35,
            alpha=0.35,
            density=True,
            label=scenario_name,
        )

    axis.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_title("Distribution Of Sell-Side Sale Ratios")
    axis.set_xlabel("Simulated Sale Price / Expected Q50")
    axis.set_ylabel("Density")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)

    figure.savefig(OUTPUT_DIR / "sale_ratio_distribution_by_scenario.png", dpi=160)
    plt.close(figure)


def write_feedback_loop_report(scenario_summary: pd.DataFrame, monthly_summary: pd.DataFrame) -> None:
    """Write a readable report explaining each stress test and its learning."""

    summary_for_report = scenario_summary.copy()
    for column in [
        "implied_market_bias",
        "average_offer_change_pct",
        "median_offer_change_pct",
    ]:
        summary_for_report[column] = summary_for_report[column].map(lambda value: f"{value:.2%}")
    for column in [
        "average_original_offer",
        "average_feedback_adjusted_offer",
        "average_offer_change",
        "median_offer_change",
    ]:
        summary_for_report[column] = summary_for_report[column].map(lambda value: f"${value:,.0f}")
    for column in ["shock_factor", "market_multiplier", "median_sale_ratio", "mean_sale_ratio"]:
        summary_for_report[column] = summary_for_report[column].map(lambda value: f"{value:.3f}")

    downside_5 = scenario_summary.set_index("scenario").loc["downside_5pct"]
    neutral = scenario_summary.set_index("scenario").loc["neutral"]
    upside = scenario_summary.set_index("scenario").loc["upside_5pct"]

    report = f"""# Sell-Side Feedback Loop Stress Test

This stress test closes the loop between the buy-side AVM and sell-side realized
outcomes.

The feedback rule is identical in every scenario:

```text
simulated_sale_price = actual_test_price * shock_factor
sale_ratio = simulated_sale_price / expected_sale_value_q50
market_multiplier = median(sale_ratio)
future_q10_q50_q90 = current_q10_q50_q90 * market_multiplier
future_offer = offer_engine(future_q10_q50_q90)
```

The shock scenario changes only the observed sell-side outcome. The correction
logic is symmetric: weak sell-side outcomes lower future offers, while stronger
sell-side outcomes raise future offers.

## Scenario Summary

{summary_for_report.to_markdown(index=False)}

![Market multiplier by scenario](market_multiplier_by_scenario.png)

![Offer change by scenario](offer_change_by_scenario.png)

## What We Learn

- In the `downside_5pct` scenario, the market multiplier is `{downside_5['market_multiplier']:.3f}`, producing an average future offer change of `{downside_5['average_offer_change_pct']:.2%}`.
- In the `neutral` scenario, the multiplier is `{neutral['market_multiplier']:.3f}`. This reflects the model's observed test-period bias before any extra shock.
- In the `upside_5pct` scenario, the multiplier is `{upside['market_multiplier']:.3f}`, so the same feedback loop can raise future offers when sell-side outcomes are stronger than expected.

The project takeaway is that valuation uncertainty and market bias are handled
separately:

```text
calibrated Q10-Q90 width -> property-level uncertainty spread
sell-side market multiplier -> market-level bias correction
```

## Monthly View

The monthly view is a lightweight version of how this would operate in
production: recent sales update the multiplier that future acquisition offers
use.

![Monthly market multiplier by scenario](monthly_market_multiplier_by_scenario.png)

![Sale ratio distribution by scenario](sale_ratio_distribution_by_scenario.png)
"""

    (OUTPUT_DIR / "sell_side_feedback_loop_report.md").write_text(report)


def main() -> None:
    """Run sell-side stress scenarios and save feedback-loop diagnostics."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pricing_module = load_pricing_module()

    # Step 1: Rebuild the calibrated buy-side state from the main pricing engine.
    actual_prices, calibrated_predictions, original_offers, test_context = build_calibrated_buy_side_state(
        pricing_module
    )

    # Step 2: Stress sell-side outcomes. Each scenario uses the same feedback
    # loop, but changes the market environment by changing the shock factor.
    scenario_summary, scenario_details, monthly_summary = run_feedback_scenarios(
        actual_prices,
        calibrated_predictions,
        original_offers,
        test_context,
        pricing_module,
    )

    # Step 3: Save outputs for the deck/report.
    scenario_summary.to_csv(OUTPUT_DIR / "feedback_scenario_summary.csv", index=False)
    scenario_details.to_csv(OUTPUT_DIR / "feedback_scenario_details.csv", index=False)
    monthly_summary.to_csv(OUTPUT_DIR / "feedback_monthly_multiplier.csv", index=False)

    plot_market_multipliers(scenario_summary)
    plot_offer_changes(scenario_summary)
    plot_monthly_multipliers(monthly_summary)
    plot_sale_ratio_distribution(scenario_details)
    write_feedback_loop_report(scenario_summary, monthly_summary)

    print("\nSell-side feedback loop scenario summary:")
    print(scenario_summary.round(4).to_string(index=False))
    print(f"\nSaved feedback-loop report to {OUTPUT_DIR / 'sell_side_feedback_loop_report.md'}")


if __name__ == "__main__":
    main()
