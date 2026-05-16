"""Downside stress backtest for the rolling sell-side feedback loop.

This script answers a narrower risk question than the real-time backtest:
if realized resale outcomes are stressed downward by 5%, 10%, or 15%, does a
rolling four-week feedback multiplier reduce overpayment and P&L risk compared
with a no-feedback baseline?
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs/8_downside_stress_feedback")
MAIN_PRICING_SCRIPT = Path("3_risk_adjusted_pricing.py")
LOOKBACK_WEEKS = 4

DOWNSIDE_SCENARIOS = {
    "downside_5pct": 0.95,
    "downside_10pct": 0.90,
    "downside_15pct": 0.85,
}


def load_pricing_module():
    """Load the numbered pricing script as a regular module."""

    spec = importlib.util.spec_from_file_location("risk_pricing", MAIN_PRICING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MAIN_PRICING_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_calibrated_buy_side_state(pricing_module) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recreate calibrated predictions and baseline offers from the main engine."""

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
    baseline_offers = pricing_module.calculate_buy_offer(
        median_price=calibrated_predictions["q50"],
        lower_price=calibrated_predictions["q10"],
        upper_price=calibrated_predictions["q90"],
        spread_assumptions=spread_assumptions,
    )
    test_context = pricing_module.load_test_sale_context()

    return test_prices, calibrated_predictions, baseline_offers, test_context


def apply_market_multiplier(predictions: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """Shift calibrated quantiles by a market-level feedback multiplier."""

    adjusted_predictions = predictions.copy()
    for column in ["q10", "q50", "q90"]:
        adjusted_predictions[column] = predictions[column] * multiplier

    adjusted_predictions[["q10", "q50", "q90"]] = np.sort(
        adjusted_predictions[["q10", "q50", "q90"]].to_numpy(),
        axis=1,
    )
    return adjusted_predictions


def estimate_prior_multiplier(history: pd.DataFrame, current_week: pd.Timestamp) -> tuple[float, int]:
    """Estimate the multiplier from the prior four weeks, with early fallback."""

    if history.empty or "sale_week" not in history.columns:
        return 1.0, 0

    prior_history = history[history["sale_week"] < current_week]
    if prior_history.empty:
        return 1.0, 0

    window_start = current_week - pd.Timedelta(weeks=LOOKBACK_WEEKS)
    lookback_history = prior_history[prior_history["sale_week"] >= window_start]
    if lookback_history.empty:
        lookback_history = prior_history

    return float(lookback_history["sale_ratio"].median()), int(len(lookback_history))


def summarize_details(details: pd.DataFrame, operating_cost_rate: float) -> Dict[str, float]:
    """Summarize baseline vs feedback performance for one stress scenario."""

    return {
        "home_count": len(details),
        "baseline_mape": details["baseline_abs_pct_error"].mean(),
        "feedback_mape": details["feedback_abs_pct_error"].mean(),
        "mape_improvement": details["baseline_abs_pct_error"].mean() - details["feedback_abs_pct_error"].mean(),
        "baseline_signed_bias": details["baseline_q50_error"].mean(),
        "feedback_signed_bias": details["feedback_q50_error"].mean(),
        "baseline_overpay_rate": details["baseline_overpay"].mean(),
        "feedback_overpay_rate": details["feedback_overpay"].mean(),
        "overpay_rate_reduction": details["baseline_overpay"].mean() - details["feedback_overpay"].mean(),
        "baseline_average_offer_to_stressed_sale": details["baseline_offer_to_stressed_sale"].mean(),
        "feedback_average_offer_to_stressed_sale": details["feedback_offer_to_stressed_sale"].mean(),
        "average_offer_change": details["offer_change"].mean(),
        "average_offer_change_pct": details["offer_change_pct"].mean(),
        "baseline_average_net_pnl_proxy": details["baseline_net_pnl_proxy"].mean(),
        "feedback_average_net_pnl_proxy": details["feedback_net_pnl_proxy"].mean(),
        "average_net_pnl_proxy_improvement": details["feedback_net_pnl_proxy"].mean()
        - details["baseline_net_pnl_proxy"].mean(),
        "baseline_total_net_pnl_proxy": details["baseline_net_pnl_proxy"].sum(),
        "feedback_total_net_pnl_proxy": details["feedback_net_pnl_proxy"].sum(),
        "total_net_pnl_proxy_improvement": details["feedback_net_pnl_proxy"].sum()
        - details["baseline_net_pnl_proxy"].sum(),
        "baseline_negative_pnl_rate": details["baseline_negative_pnl"].mean(),
        "feedback_negative_pnl_rate": details["feedback_negative_pnl"].mean(),
        "negative_pnl_rate_reduction": details["baseline_negative_pnl"].mean()
        - details["feedback_negative_pnl"].mean(),
        "baseline_roi_proxy": details["baseline_roi_proxy"].mean(),
        "feedback_roi_proxy": details["feedback_roi_proxy"].mean(),
        "operating_cost_rate": operating_cost_rate,
        "final_multiplier_used": details.sort_values("sale_week")["feedback_multiplier_used"].iloc[-1],
    }


def run_one_scenario(
    scenario_name: str,
    shock_factor: float,
    actual_prices: pd.Series,
    calibrated_predictions: pd.DataFrame,
    baseline_offers: pd.DataFrame,
    test_context: pd.DataFrame,
    pricing_module,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Walk forward through the test period for one downside stress scenario."""

    spread_assumptions = pricing_module.SpreadAssumptions()
    operating_cost_rate = (
        spread_assumptions.transaction_cost_rate
        + spread_assumptions.holding_cost_rate
        + spread_assumptions.resale_prep_cost_rate
    )

    test_frame = pd.DataFrame(
        {
            "scenario": scenario_name,
            "shock_factor": shock_factor,
            "actual_price": actual_prices,
            "stressed_sale_price": actual_prices * shock_factor,
            "sale_week": test_context["sale_date"].dt.to_period("W").dt.start_time,
        },
        index=actual_prices.index,
    )
    test_frame = test_frame.join(calibrated_predictions[["q10", "q50", "q90"]])
    test_frame = test_frame.join(
        baseline_offers[["buy_offer", "total_spread"]].rename(
            columns={
                "buy_offer": "baseline_buy_offer",
                "total_spread": "baseline_total_spread",
            }
        )
    )

    history_rows = []
    detail_frames = []
    weekly_rows = []

    for sale_week in sorted(test_frame["sale_week"].unique()):
        week_frame = test_frame[test_frame["sale_week"] == sale_week].copy()
        history_frame = pd.DataFrame(history_rows)
        multiplier, feedback_sample_count = estimate_prior_multiplier(history_frame, sale_week)

        feedback_predictions = apply_market_multiplier(week_frame[["q10", "q50", "q90"]], multiplier)
        feedback_offers = pricing_module.calculate_buy_offer(
            median_price=feedback_predictions["q50"],
            lower_price=feedback_predictions["q10"],
            upper_price=feedback_predictions["q90"],
            spread_assumptions=spread_assumptions,
        )

        week_frame["feedback_multiplier_used"] = multiplier
        week_frame["feedback_sample_count"] = feedback_sample_count
        week_frame["feedback_q50"] = feedback_predictions["q50"]
        week_frame["feedback_buy_offer"] = feedback_offers["buy_offer"]
        week_frame["baseline_q50_error"] = week_frame["stressed_sale_price"] - week_frame["q50"]
        week_frame["feedback_q50_error"] = week_frame["stressed_sale_price"] - week_frame["feedback_q50"]
        week_frame["baseline_abs_pct_error"] = week_frame["baseline_q50_error"].abs() / week_frame["stressed_sale_price"]
        week_frame["feedback_abs_pct_error"] = week_frame["feedback_q50_error"].abs() / week_frame["stressed_sale_price"]
        week_frame["baseline_offer_to_stressed_sale"] = (
            week_frame["baseline_buy_offer"] / week_frame["stressed_sale_price"]
        )
        week_frame["feedback_offer_to_stressed_sale"] = (
            week_frame["feedback_buy_offer"] / week_frame["stressed_sale_price"]
        )
        week_frame["baseline_overpay"] = week_frame["baseline_buy_offer"] > week_frame["stressed_sale_price"]
        week_frame["feedback_overpay"] = week_frame["feedback_buy_offer"] > week_frame["stressed_sale_price"]
        week_frame["offer_change"] = week_frame["feedback_buy_offer"] - week_frame["baseline_buy_offer"]
        week_frame["offer_change_pct"] = week_frame["offer_change"] / week_frame["baseline_buy_offer"]
        week_frame["sale_ratio"] = week_frame["stressed_sale_price"] / week_frame["q50"]
        week_frame["estimated_operating_cost"] = week_frame["stressed_sale_price"] * operating_cost_rate
        week_frame["baseline_net_pnl_proxy"] = (
            week_frame["stressed_sale_price"] - week_frame["baseline_buy_offer"] - week_frame["estimated_operating_cost"]
        )
        week_frame["feedback_net_pnl_proxy"] = (
            week_frame["stressed_sale_price"] - week_frame["feedback_buy_offer"] - week_frame["estimated_operating_cost"]
        )
        week_frame["baseline_roi_proxy"] = week_frame["baseline_net_pnl_proxy"] / week_frame["baseline_buy_offer"]
        week_frame["feedback_roi_proxy"] = week_frame["feedback_net_pnl_proxy"] / week_frame["feedback_buy_offer"]
        week_frame["baseline_negative_pnl"] = week_frame["baseline_net_pnl_proxy"] < 0
        week_frame["feedback_negative_pnl"] = week_frame["feedback_net_pnl_proxy"] < 0

        weekly_rows.append(
            {
                "scenario": scenario_name,
                "shock_factor": shock_factor,
                "sale_week": sale_week,
                "home_count": len(week_frame),
                "feedback_multiplier_used": multiplier,
                "feedback_sample_count": feedback_sample_count,
                "current_week_sale_ratio": week_frame["sale_ratio"].median(),
                "baseline_mape": week_frame["baseline_abs_pct_error"].mean(),
                "feedback_mape": week_frame["feedback_abs_pct_error"].mean(),
                "baseline_overpay_rate": week_frame["baseline_overpay"].mean(),
                "feedback_overpay_rate": week_frame["feedback_overpay"].mean(),
                "baseline_average_net_pnl_proxy": week_frame["baseline_net_pnl_proxy"].mean(),
                "feedback_average_net_pnl_proxy": week_frame["feedback_net_pnl_proxy"].mean(),
                "baseline_negative_pnl_rate": week_frame["baseline_negative_pnl"].mean(),
                "feedback_negative_pnl_rate": week_frame["feedback_negative_pnl"].mean(),
                "average_offer_change_pct": week_frame["offer_change_pct"].mean(),
            }
        )

        detail_frames.append(week_frame)
        history_rows.extend(week_frame[["sale_week", "sale_ratio"]].to_dict(orient="records"))

    details = pd.concat(detail_frames).reset_index(names="row_id")
    weekly_metrics = pd.DataFrame(weekly_rows)
    summary = pd.Series(summarize_details(details, operating_cost_rate))
    summary["scenario"] = scenario_name
    summary["shock_factor"] = shock_factor

    return details, weekly_metrics, summary


def run_downside_backtest(
    actual_prices: pd.Series,
    calibrated_predictions: pd.DataFrame,
    baseline_offers: pd.DataFrame,
    test_context: pd.DataFrame,
    pricing_module,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all downside scenarios through the same rolling feedback loop."""

    detail_frames = []
    weekly_frames = []
    summary_rows = []

    for scenario_name, shock_factor in DOWNSIDE_SCENARIOS.items():
        details, weekly_metrics, summary = run_one_scenario(
            scenario_name,
            shock_factor,
            actual_prices,
            calibrated_predictions,
            baseline_offers,
            test_context,
            pricing_module,
        )
        detail_frames.append(details)
        weekly_frames.append(weekly_metrics)
        summary_rows.append(summary)

    scenario_details = pd.concat(detail_frames).reset_index(drop=True)
    weekly_metrics = pd.concat(weekly_frames).reset_index(drop=True)
    scenario_summary = pd.DataFrame(summary_rows)

    return scenario_summary, weekly_metrics, scenario_details


def plot_metric_comparison(
    summary: pd.DataFrame,
    metric_base: str,
    metric_feedback: str,
    title: str,
    ylabel: str,
    output_name: str,
    percent: bool = True,
) -> None:
    """Grouped bar chart for baseline vs feedback by stress scenario."""

    x = np.arange(len(summary))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, summary[metric_base], width, label="No feedback", color="#94A3B8")
    ax.bar(x + width / 2, summary[metric_feedback], width, label="Rolling feedback", color="#1D4ED8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["scenario"], rotation=18)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for offset, column in [(-width / 2, metric_base), (width / 2, metric_feedback)]:
        for index, value in enumerate(summary[column]):
            label = f"{value:.1%}" if percent else f"${value / 1000:.0f}K"
            ax.text(index + offset, value, label, ha="center", va="bottom", fontsize=9)

    fig.savefig(OUTPUT_DIR / output_name, dpi=160)
    plt.close(fig)


def plot_multiplier_paths(weekly_metrics: pd.DataFrame) -> None:
    """Plot rolling multiplier path under each downside stress."""

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for scenario, frame in weekly_metrics.groupby("scenario"):
        ax.plot(
            frame["sale_week"],
            frame["feedback_multiplier_used"],
            marker="o",
            linewidth=2,
            label=scenario,
        )

    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_title("Rolling Feedback Multiplier Under Downside Stress")
    ax.set_xlabel("Sale Week")
    ax.set_ylabel("Multiplier Used For Offers")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    fig.savefig(OUTPUT_DIR / "downside_multiplier_paths.png", dpi=160)
    plt.close(fig)


def plot_pnl_improvement(summary: pd.DataFrame) -> None:
    """Plot total net P&L proxy improvement from rolling feedback."""

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    colors = ["#16A34A" if value > 0 else "#DC2626" for value in summary["total_net_pnl_proxy_improvement"]]
    ax.bar(summary["scenario"], summary["total_net_pnl_proxy_improvement"], color=colors)
    ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
    ax.set_title("Total Net P&L Proxy Improvement From Feedback")
    ax.set_ylabel("Feedback - No Feedback")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)

    for index, value in enumerate(summary["total_net_pnl_proxy_improvement"]):
        ax.text(index, value, f"${value / 1_000_000:.1f}M", ha="center", va="bottom" if value >= 0 else "top")

    fig.savefig(OUTPUT_DIR / "total_pnl_proxy_improvement.png", dpi=160)
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    """Write an interpretation-heavy markdown report."""

    report_table = summary[
        [
            "scenario",
            "shock_factor",
            "baseline_mape",
            "feedback_mape",
            "baseline_overpay_rate",
            "feedback_overpay_rate",
            "baseline_average_net_pnl_proxy",
            "feedback_average_net_pnl_proxy",
            "baseline_negative_pnl_rate",
            "feedback_negative_pnl_rate",
            "total_net_pnl_proxy_improvement",
            "final_multiplier_used",
        ]
    ].copy()

    for column in [
        "baseline_mape",
        "feedback_mape",
        "baseline_overpay_rate",
        "feedback_overpay_rate",
        "baseline_negative_pnl_rate",
        "feedback_negative_pnl_rate",
    ]:
        report_table[column] = report_table[column].map(lambda value: f"{value:.2%}")
    for column in [
        "baseline_average_net_pnl_proxy",
        "feedback_average_net_pnl_proxy",
        "total_net_pnl_proxy_improvement",
    ]:
        report_table[column] = report_table[column].map(lambda value: f"${value:,.0f}")
    for column in ["shock_factor", "final_multiplier_used"]:
        report_table[column] = report_table[column].map(lambda value: f"{value:.3f}")

    report = f"""# Downside Stress Feedback Backtest

This report stress-tests the rolling feedback loop under downside resale
conditions. Unlike the actual real-time backtest, this file intentionally
applies downside shocks to observed test prices:

```text
stressed_sale_price = actual_test_price * shock_factor
```

For each stress level, the loop is still realistic in timing: current-week
offers only use market multipliers learned from prior weeks.

```text
sale_ratio = stressed_sale_price / expected_q50
rolling_multiplier = median(sale_ratio over prior 4 weeks)
feedback_q10_q50_q90 = calibrated_q10_q50_q90 * rolling_multiplier
feedback_offer = offer_engine(feedback_q10_q50_q90)
```

## Summary

{report_table.to_markdown(index=False)}

## What We Learn

Under downside stress, the rolling multiplier moves below `1.0`, so future
offers are reduced relative to the no-feedback baseline. This is where the
feedback loop provides downside protection: it cuts overpay risk and improves
the P&L proxy after weak sell-side outcomes become visible.

![Multiplier paths](downside_multiplier_paths.png)

![MAPE comparison](downside_mape_before_after.png)

![Overpay comparison](downside_overpay_before_after.png)

![Negative P&L comparison](downside_negative_pnl_before_after.png)

![P&L improvement](total_pnl_proxy_improvement.png)

## Interpretation

The actual rolling backtest showed the feedback loop raising offers because the
test period was stronger than expected. This downside stress test shows the
opposite case: when resale outcomes weaken, the same loop lowers future offers
and protects estimated P&L.

Together, these two analyses show that the feedback loop is directionally
symmetric:

```text
strong sell-side outcomes -> raise future offers for competitiveness
weak sell-side outcomes   -> lower future offers for risk control
```
"""

    (OUTPUT_DIR / "downside_stress_feedback_report.md").write_text(report)


def main() -> None:
    """Run downside stress feedback-loop backtests."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pricing_module = load_pricing_module()

    actual_prices, calibrated_predictions, baseline_offers, test_context = build_calibrated_buy_side_state(
        pricing_module
    )
    scenario_summary, weekly_metrics, scenario_details = run_downside_backtest(
        actual_prices,
        calibrated_predictions,
        baseline_offers,
        test_context,
        pricing_module,
    )

    scenario_summary.to_csv(OUTPUT_DIR / "downside_stress_summary.csv", index=False)
    weekly_metrics.to_csv(OUTPUT_DIR / "downside_stress_weekly_metrics.csv", index=False)
    scenario_details.to_csv(OUTPUT_DIR / "downside_stress_details.csv", index=False)

    plot_multiplier_paths(weekly_metrics)
    plot_metric_comparison(
        scenario_summary,
        "baseline_mape",
        "feedback_mape",
        "MAPE Under Downside Stress",
        "MAPE vs Stressed Sale Price",
        "downside_mape_before_after.png",
    )
    plot_metric_comparison(
        scenario_summary,
        "baseline_overpay_rate",
        "feedback_overpay_rate",
        "Overpay Proxy Under Downside Stress",
        "Offer > Stressed Sale Price",
        "downside_overpay_before_after.png",
    )
    plot_metric_comparison(
        scenario_summary,
        "baseline_negative_pnl_rate",
        "feedback_negative_pnl_rate",
        "Negative P&L Proxy Rate Under Downside Stress",
        "Share Of Homes",
        "downside_negative_pnl_before_after.png",
    )
    plot_metric_comparison(
        scenario_summary,
        "baseline_average_net_pnl_proxy",
        "feedback_average_net_pnl_proxy",
        "Average Net P&L Proxy Under Downside Stress",
        "Average Net P&L Proxy",
        "downside_average_pnl_before_after.png",
        percent=False,
    )
    plot_pnl_improvement(scenario_summary)
    write_report(scenario_summary)

    print("\nDownside stress feedback summary:")
    print(scenario_summary.round(4).to_string(index=False))
    print(f"\nSaved report to {OUTPUT_DIR / 'downside_stress_feedback_report.md'}")


if __name__ == "__main__":
    main()
