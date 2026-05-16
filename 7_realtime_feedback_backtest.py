"""Real-time sell-side feedback backtest using actual held-out test sales.

This is the non-simulated version of the feedback loop. It walks through the
test period week by week, uses only prior resale outcomes to estimate a market
multiplier, and compares adjusted future offers against a no-feedback baseline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs/7_realtime_feedback")
MAIN_PRICING_SCRIPT = Path("3_risk_adjusted_pricing.py")
LOOKBACK_WEEKS = 4


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


def run_realtime_feedback_backtest(
    actual_prices: pd.Series,
    calibrated_predictions: pd.DataFrame,
    baseline_offers: pd.DataFrame,
    test_context: pd.DataFrame,
    pricing_module,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Walk forward through test weeks and apply only prior feedback."""

    spread_assumptions = pricing_module.SpreadAssumptions()
    operating_cost_rate = (
        spread_assumptions.transaction_cost_rate
        + spread_assumptions.holding_cost_rate
        + spread_assumptions.resale_prep_cost_rate
    )
    test_frame = pd.DataFrame(
        {
            "actual_price": actual_prices,
            "sale_week": test_context["sale_date"].dt.to_period("W").dt.start_time,
        },
        index=actual_prices.index,
    )
    test_frame = test_frame.join(calibrated_predictions[["q10", "q50", "q90"]])
    test_frame = test_frame.join(
        baseline_offers[
            [
                "buy_offer",
                "uncertainty_score",
                "total_spread",
                "hit_max_uncertainty_penalty",
            ]
        ].rename(
            columns={
                "buy_offer": "baseline_buy_offer",
                "uncertainty_score": "baseline_uncertainty_score",
                "total_spread": "baseline_total_spread",
                "hit_max_uncertainty_penalty": "baseline_hit_cap",
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
        week_frame["feedback_total_spread"] = feedback_offers["total_spread"]

        week_frame["baseline_q50_error"] = week_frame["actual_price"] - week_frame["q50"]
        week_frame["feedback_q50_error"] = week_frame["actual_price"] - week_frame["feedback_q50"]
        week_frame["baseline_abs_pct_error"] = week_frame["baseline_q50_error"].abs() / week_frame["actual_price"]
        week_frame["feedback_abs_pct_error"] = week_frame["feedback_q50_error"].abs() / week_frame["actual_price"]
        week_frame["baseline_offer_to_actual"] = week_frame["baseline_buy_offer"] / week_frame["actual_price"]
        week_frame["feedback_offer_to_actual"] = week_frame["feedback_buy_offer"] / week_frame["actual_price"]
        week_frame["baseline_overpay"] = week_frame["baseline_buy_offer"] > week_frame["actual_price"]
        week_frame["feedback_overpay"] = week_frame["feedback_buy_offer"] > week_frame["actual_price"]
        week_frame["offer_change"] = week_frame["feedback_buy_offer"] - week_frame["baseline_buy_offer"]
        week_frame["offer_change_pct"] = week_frame["offer_change"] / week_frame["baseline_buy_offer"]
        week_frame["sale_ratio"] = week_frame["actual_price"] / week_frame["q50"]
        week_frame["estimated_operating_cost"] = week_frame["actual_price"] * operating_cost_rate
        week_frame["baseline_gross_pnl"] = week_frame["actual_price"] - week_frame["baseline_buy_offer"]
        week_frame["feedback_gross_pnl"] = week_frame["actual_price"] - week_frame["feedback_buy_offer"]
        week_frame["baseline_net_pnl_proxy"] = (
            week_frame["baseline_gross_pnl"] - week_frame["estimated_operating_cost"]
        )
        week_frame["feedback_net_pnl_proxy"] = (
            week_frame["feedback_gross_pnl"] - week_frame["estimated_operating_cost"]
        )
        week_frame["baseline_roi_proxy"] = week_frame["baseline_net_pnl_proxy"] / week_frame["baseline_buy_offer"]
        week_frame["feedback_roi_proxy"] = week_frame["feedback_net_pnl_proxy"] / week_frame["feedback_buy_offer"]
        week_frame["baseline_negative_pnl"] = week_frame["baseline_net_pnl_proxy"] < 0
        week_frame["feedback_negative_pnl"] = week_frame["feedback_net_pnl_proxy"] < 0

        weekly_rows.append(
            {
                "sale_week": sale_week,
                "home_count": len(week_frame),
                "feedback_multiplier_used": multiplier,
                "feedback_sample_count": feedback_sample_count,
                "current_week_sale_ratio": week_frame["sale_ratio"].median(),
                "baseline_mape": week_frame["baseline_abs_pct_error"].mean(),
                "feedback_mape": week_frame["feedback_abs_pct_error"].mean(),
                "baseline_signed_bias": week_frame["baseline_q50_error"].mean(),
                "feedback_signed_bias": week_frame["feedback_q50_error"].mean(),
                "baseline_overpay_rate": week_frame["baseline_overpay"].mean(),
                "feedback_overpay_rate": week_frame["feedback_overpay"].mean(),
                "average_baseline_offer_to_actual": week_frame["baseline_offer_to_actual"].mean(),
                "average_feedback_offer_to_actual": week_frame["feedback_offer_to_actual"].mean(),
                "average_offer_change": week_frame["offer_change"].mean(),
                "average_offer_change_pct": week_frame["offer_change_pct"].mean(),
                "baseline_average_net_pnl_proxy": week_frame["baseline_net_pnl_proxy"].mean(),
                "feedback_average_net_pnl_proxy": week_frame["feedback_net_pnl_proxy"].mean(),
                "baseline_total_net_pnl_proxy": week_frame["baseline_net_pnl_proxy"].sum(),
                "feedback_total_net_pnl_proxy": week_frame["feedback_net_pnl_proxy"].sum(),
                "baseline_roi_proxy": week_frame["baseline_roi_proxy"].mean(),
                "feedback_roi_proxy": week_frame["feedback_roi_proxy"].mean(),
                "baseline_negative_pnl_rate": week_frame["baseline_negative_pnl"].mean(),
                "feedback_negative_pnl_rate": week_frame["feedback_negative_pnl"].mean(),
            }
        )

        detail_frames.append(week_frame)
        history_rows.extend(
            week_frame[["sale_week", "sale_ratio"]].to_dict(orient="records")
        )

    details = pd.concat(detail_frames).reset_index(names="row_id")
    weekly_metrics = pd.DataFrame(weekly_rows)

    summary = pd.DataFrame(
        [
            {
                "baseline_mape": details["baseline_abs_pct_error"].mean(),
                "feedback_mape": details["feedback_abs_pct_error"].mean(),
                "mape_improvement": details["baseline_abs_pct_error"].mean()
                - details["feedback_abs_pct_error"].mean(),
                "baseline_median_abs_pct_error": details["baseline_abs_pct_error"].median(),
                "feedback_median_abs_pct_error": details["feedback_abs_pct_error"].median(),
                "baseline_signed_bias": details["baseline_q50_error"].mean(),
                "feedback_signed_bias": details["feedback_q50_error"].mean(),
                "baseline_overpay_rate": details["baseline_overpay"].mean(),
                "feedback_overpay_rate": details["feedback_overpay"].mean(),
                "baseline_average_offer_to_actual": details["baseline_offer_to_actual"].mean(),
                "feedback_average_offer_to_actual": details["feedback_offer_to_actual"].mean(),
                "average_offer_change": details["offer_change"].mean(),
                "average_offer_change_pct": details["offer_change_pct"].mean(),
                "baseline_average_net_pnl_proxy": details["baseline_net_pnl_proxy"].mean(),
                "feedback_average_net_pnl_proxy": details["feedback_net_pnl_proxy"].mean(),
                "baseline_median_net_pnl_proxy": details["baseline_net_pnl_proxy"].median(),
                "feedback_median_net_pnl_proxy": details["feedback_net_pnl_proxy"].median(),
                "baseline_total_net_pnl_proxy": details["baseline_net_pnl_proxy"].sum(),
                "feedback_total_net_pnl_proxy": details["feedback_net_pnl_proxy"].sum(),
                "pnl_proxy_change": details["feedback_net_pnl_proxy"].sum()
                - details["baseline_net_pnl_proxy"].sum(),
                "baseline_roi_proxy": details["baseline_roi_proxy"].mean(),
                "feedback_roi_proxy": details["feedback_roi_proxy"].mean(),
                "baseline_negative_pnl_rate": details["baseline_negative_pnl"].mean(),
                "feedback_negative_pnl_rate": details["feedback_negative_pnl"].mean(),
                "operating_cost_rate": operating_cost_rate,
                "final_multiplier_used": weekly_metrics["feedback_multiplier_used"].iloc[-1],
            }
        ]
    )

    return summary, weekly_metrics, details


def plot_multiplier_path(weekly_metrics: pd.DataFrame) -> None:
    """Plot the multiplier used for each week's offers."""

    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    axis.plot(
        weekly_metrics["sale_week"],
        weekly_metrics["feedback_multiplier_used"],
        marker="o",
        linewidth=2.5,
        color="#1D4ED8",
        label="Multiplier used",
    )
    axis.plot(
        weekly_metrics["sale_week"],
        weekly_metrics["current_week_sale_ratio"],
        marker="o",
        linewidth=1.8,
        color="#BE185D",
        alpha=0.75,
        label="Current week observed ratio",
    )
    axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_title("Real-Time Feedback Multiplier Path")
    axis.set_xlabel("Sale Week")
    axis.set_ylabel("Sale Ratio / Multiplier")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)

    figure.savefig(OUTPUT_DIR / "rolling_multiplier_over_time.png", dpi=160)
    plt.close(figure)


def plot_error_comparison(summary: pd.DataFrame) -> None:
    """Compare baseline vs feedback valuation error."""

    values = [
        summary.loc[0, "baseline_mape"],
        summary.loc[0, "feedback_mape"],
    ]
    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    axis.bar(["No Feedback", "Rolling Feedback"], values, color=["#94A3B8", "#1D4ED8"])
    axis.set_title("Valuation Error Before vs After Feedback")
    axis.set_ylabel("MAPE vs Actual Test Price")
    axis.set_ylim(0, max(values) * 1.25)
    axis.grid(axis="y", alpha=0.25)

    for index, value in enumerate(values):
        axis.text(index, value + 0.003, f"{value:.2%}", ha="center", fontsize=10)

    figure.savefig(OUTPUT_DIR / "valuation_error_before_after.png", dpi=160)
    plt.close(figure)


def plot_weekly_bias(weekly_metrics: pd.DataFrame) -> None:
    """Plot weekly signed bias before and after feedback."""

    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    axis.plot(
        weekly_metrics["sale_week"],
        weekly_metrics["baseline_signed_bias"],
        marker="o",
        linewidth=2,
        color="#64748B",
        label="No feedback",
    )
    axis.plot(
        weekly_metrics["sale_week"],
        weekly_metrics["feedback_signed_bias"],
        marker="o",
        linewidth=2,
        color="#1D4ED8",
        label="Rolling feedback",
    )
    axis.axhline(0, color="#333333", linestyle="--", linewidth=1)
    axis.set_title("Weekly Valuation Bias Before vs After Feedback")
    axis.set_xlabel("Sale Week")
    axis.set_ylabel("Actual Price - Expected Q50")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)

    figure.savefig(OUTPUT_DIR / "weekly_bias_before_after.png", dpi=160)
    plt.close(figure)


def plot_offer_to_actual_distribution(details: pd.DataFrame) -> None:
    """Compare offer-to-actual-price distributions."""

    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.hist(
        details["baseline_offer_to_actual"],
        bins=40,
        alpha=0.50,
        density=True,
        color="#94A3B8",
        label="No feedback",
    )
    axis.hist(
        details["feedback_offer_to_actual"],
        bins=40,
        alpha=0.45,
        density=True,
        color="#1D4ED8",
        label="Rolling feedback",
    )
    axis.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_title("Offer-To-Actual Distribution")
    axis.set_xlabel("Buy Offer / Actual Sale Price")
    axis.set_ylabel("Density")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)

    figure.savefig(OUTPUT_DIR / "offer_to_actual_distribution.png", dpi=160)
    plt.close(figure)


def plot_offer_change_over_time(weekly_metrics: pd.DataFrame) -> None:
    """Show how rolling feedback changes customer-facing offers over time."""

    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    axis.bar(
        weekly_metrics["sale_week"],
        weekly_metrics["average_offer_change_pct"],
        color="#1D4ED8",
        alpha=0.85,
    )
    axis.axhline(0, color="#333333", linestyle="--", linewidth=1)
    axis.set_title("Average Offer Change From Rolling Feedback")
    axis.set_xlabel("Sale Week")
    axis.set_ylabel("Feedback Offer Change")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)

    figure.savefig(OUTPUT_DIR / "offer_change_over_time.png", dpi=160)
    plt.close(figure)


def plot_overpay_rate_comparison(summary: pd.DataFrame) -> None:
    """Compare overpay-rate proxy before and after feedback."""

    values = [
        summary.loc[0, "baseline_overpay_rate"],
        summary.loc[0, "feedback_overpay_rate"],
    ]
    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    axis.bar(["No Feedback", "Rolling Feedback"], values, color=["#94A3B8", "#1D4ED8"])
    axis.set_title("Overpay-Rate Proxy Before vs After Feedback")
    axis.set_ylabel("Share With Offer > Actual Test Price")
    axis.set_ylim(0, max(values) * 1.35 if max(values) > 0 else 0.05)
    axis.grid(axis="y", alpha=0.25)

    for index, value in enumerate(values):
        axis.text(index, value + 0.002, f"{value:.2%}", ha="center", fontsize=10)

    figure.savefig(OUTPUT_DIR / "overpay_rate_before_after.png", dpi=160)
    plt.close(figure)


def plot_pnl_comparison(summary: pd.DataFrame) -> None:
    """Compare average net P&L proxy before and after feedback."""

    values = [
        summary.loc[0, "baseline_average_net_pnl_proxy"],
        summary.loc[0, "feedback_average_net_pnl_proxy"],
    ]
    figure, axis = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    axis.bar(["No Feedback", "Rolling Feedback"], values, color=["#94A3B8", "#1D4ED8"])
    axis.axhline(0, color="#333333", linestyle="--", linewidth=1)
    axis.set_title("Average Net P&L Proxy Before vs After Feedback")
    axis.set_ylabel("Average Net P&L Proxy Per Home")
    axis.grid(axis="y", alpha=0.25)

    for index, value in enumerate(values):
        offset = max(abs(max(values)), abs(min(values)), 1) * 0.03
        axis.text(index, value + offset, f"${value:,.0f}", ha="center", fontsize=10)

    figure.savefig(OUTPUT_DIR / "pnl_proxy_before_after.png", dpi=160)
    plt.close(figure)


def plot_weekly_pnl(weekly_metrics: pd.DataFrame) -> None:
    """Plot weekly average net P&L proxy before and after feedback."""

    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    axis.plot(
        weekly_metrics["sale_week"],
        weekly_metrics["baseline_average_net_pnl_proxy"],
        marker="o",
        linewidth=2,
        color="#64748B",
        label="No feedback",
    )
    axis.plot(
        weekly_metrics["sale_week"],
        weekly_metrics["feedback_average_net_pnl_proxy"],
        marker="o",
        linewidth=2,
        color="#1D4ED8",
        label="Rolling feedback",
    )
    axis.axhline(0, color="#333333", linestyle="--", linewidth=1)
    axis.set_title("Weekly Average Net P&L Proxy")
    axis.set_xlabel("Sale Week")
    axis.set_ylabel("Average Net P&L Proxy")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)

    figure.savefig(OUTPUT_DIR / "weekly_pnl_proxy_before_after.png", dpi=160)
    plt.close(figure)


def plot_negative_pnl_rate(summary: pd.DataFrame) -> None:
    """Compare the share of homes with negative net P&L proxy."""

    values = [
        summary.loc[0, "baseline_negative_pnl_rate"],
        summary.loc[0, "feedback_negative_pnl_rate"],
    ]
    figure, axis = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    axis.bar(["No Feedback", "Rolling Feedback"], values, color=["#94A3B8", "#1D4ED8"])
    axis.set_title("Negative Net P&L Proxy Rate")
    axis.set_ylabel("Share Of Homes")
    axis.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 0.05)
    axis.grid(axis="y", alpha=0.25)

    for index, value in enumerate(values):
        axis.text(index, value + 0.003, f"{value:.2%}", ha="center", fontsize=10)

    figure.savefig(OUTPUT_DIR / "negative_pnl_rate_before_after.png", dpi=160)
    plt.close(figure)


def write_report(summary: pd.DataFrame, weekly_metrics: pd.DataFrame) -> None:
    """Write an explanation-heavy markdown report for the rolling backtest."""

    row = summary.iloc[0]
    first_week = weekly_metrics["sale_week"].min().date()
    last_week = weekly_metrics["sale_week"].max().date()

    report = f"""# Real-Time Feedback Backtest

This backtest uses the actual held-out test period only. There are no shock
factors and no simulated market scenarios.

The test period runs from `{first_week}` to `{last_week}`. For each sale week,
the feedback system uses a market multiplier estimated only from prior resale
outcomes, so the current week is never allowed to look at its own actuals.

## Feedback Rule

```text
sale_ratio = actual_test_price / expected_q50
rolling_multiplier = median(sale_ratio over prior 4 weeks)
feedback_q10_q50_q90 = calibrated_q10_q50_q90 * rolling_multiplier
feedback_offer = offer_engine(feedback_q10_q50_q90)
```

For the first week, no prior sell-side outcomes exist, so the multiplier starts
at `1.0`. After that, it uses a four-week lookback. This is more stable than
using only the previous week, but still responsive to recent market movement.

## Headline Results

| Metric | No Feedback | Rolling Feedback |
|---|---:|---:|
| MAPE vs actual price | {row['baseline_mape']:.2%} | {row['feedback_mape']:.2%} |
| Median APE vs actual price | {row['baseline_median_abs_pct_error']:.2%} | {row['feedback_median_abs_pct_error']:.2%} |
| Signed bias | ${row['baseline_signed_bias']:,.0f} | ${row['feedback_signed_bias']:,.0f} |
| Offer / actual price | {row['baseline_average_offer_to_actual']:.2%} | {row['feedback_average_offer_to_actual']:.2%} |
| Overpay-rate proxy | {row['baseline_overpay_rate']:.2%} | {row['feedback_overpay_rate']:.2%} |
| Avg net P&L proxy | ${row['baseline_average_net_pnl_proxy']:,.0f} | ${row['feedback_average_net_pnl_proxy']:,.0f} |
| ROI proxy | {row['baseline_roi_proxy']:.2%} | {row['feedback_roi_proxy']:.2%} |
| Negative P&L proxy rate | {row['baseline_negative_pnl_rate']:.2%} | {row['feedback_negative_pnl_rate']:.2%} |

Average offer change from feedback:

```text
${row['average_offer_change']:,.0f} per home ({row['average_offer_change_pct']:.2%})
```

Final multiplier used:

```text
{row['final_multiplier_used']:.3f}
```

## What Happened

The held-out test period was stronger than the model expected, so the feedback
multiplier usually moves above `1.0`. That means the loop raises future offers
instead of lowering them. This is the desired behavior: the loop is correcting
observed market bias, not blindly making offers more conservative.

![Rolling multiplier over time](rolling_multiplier_over_time.png)

![Valuation error before after](valuation_error_before_after.png)

![Weekly bias before after](weekly_bias_before_after.png)

## Business Impact

The rolling feedback loop changes the customer-facing offer by shifting the
whole calibrated valuation band. It does not change the property-level
uncertainty score directly.

![Offer change over time](offer_change_over_time.png)

![Offer to actual distribution](offer_to_actual_distribution.png)

![Overpay rate before after](overpay_rate_before_after.png)

## P&L Proxy

Because the King County dataset does not include realized renovation costs,
holding period, financing costs, concessions, or resale transaction costs, this
is not true acquisition P&L. It is a proxy based on the cost assumptions already used
in the offer engine.

```text
operating_cost_rate = transaction cost + holding cost + resale prep
                    = {row['operating_cost_rate']:.2%}

net_pnl_proxy = actual_sale_price - buy_offer - actual_sale_price * operating_cost_rate
roi_proxy = net_pnl_proxy / buy_offer
```

Target margin is intentionally not treated as a cost. It is the desired profit
cushion.

P&L proxy change from feedback:

```text
Total net P&L proxy change: ${row['pnl_proxy_change']:,.0f}
```

![P&L proxy before after](pnl_proxy_before_after.png)

![Weekly P&L proxy before after](weekly_pnl_proxy_before_after.png)

![Negative P&L rate before after](negative_pnl_rate_before_after.png)

## Interpretation

In this actual test period, feedback mainly improves competitiveness and market
alignment because recent sales are clearing above the model's original Q50.
The tradeoff is visible in the P&L proxy: raising offers reduces per-home
cushion and increases negative-P&L risk. In a weaker period, the same mechanism
would push the multiplier below `1.0` and lower future offers.

This gives the project two complementary feedback-loop views:

```text
6_sell_side_feedback_loop.py       -> stress-test scenarios
7_realtime_feedback_backtest.py    -> actual rolling test-period backtest
```
"""

    (OUTPUT_DIR / "realtime_feedback_report.md").write_text(report)


def main() -> None:
    """Run the actual rolling feedback-loop backtest."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pricing_module = load_pricing_module()

    actual_prices, calibrated_predictions, baseline_offers, test_context = build_calibrated_buy_side_state(
        pricing_module
    )
    summary, weekly_metrics, details = run_realtime_feedback_backtest(
        actual_prices,
        calibrated_predictions,
        baseline_offers,
        test_context,
        pricing_module,
    )

    summary.to_csv(OUTPUT_DIR / "realtime_feedback_summary.csv", index=False)
    weekly_metrics.to_csv(OUTPUT_DIR / "weekly_multiplier_path.csv", index=False)
    details.to_csv(OUTPUT_DIR / "realtime_feedback_details.csv", index=False)

    plot_multiplier_path(weekly_metrics)
    plot_error_comparison(summary)
    plot_weekly_bias(weekly_metrics)
    plot_offer_to_actual_distribution(details)
    plot_offer_change_over_time(weekly_metrics)
    plot_overpay_rate_comparison(summary)
    plot_pnl_comparison(summary)
    plot_weekly_pnl(weekly_metrics)
    plot_negative_pnl_rate(summary)
    write_report(summary, weekly_metrics)

    print("\nReal-time feedback backtest summary:")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved report to {OUTPUT_DIR / 'realtime_feedback_report.md'}")


if __name__ == "__main__":
    main()
