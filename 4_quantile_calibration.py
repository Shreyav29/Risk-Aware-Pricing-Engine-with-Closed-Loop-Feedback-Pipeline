"""Validation-based calibration for the quantile AVM intervals.

This script learns a simple interval-widening factor on the validation period
and applies it to the untouched test period. The goal is to make Q10-Q90 behave
more like an 80% prediction interval before using interval width in offer logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from risk_adjusted_pricing import (
    SpreadAssumptions,
    calculate_buy_offer,
    load_modeling_data,
    make_quantile_model,
)


OUTPUT_DIR = Path("outputs/4_quantile_calibration")
TARGET_COVERAGE = 0.80
CALIBRATION_FACTORS = np.arange(1.00, 2.55, 0.05)


def load_best_quantile_params() -> Dict[str, object]:
    """Reuse the best quantile hyperparameters from the main AVM run."""

    tuning_path = Path("outputs/3_model_diagnostics/lightgbm_tuning_results.csv")
    if not tuning_path.exists():
        tuning_path = Path("outputs/lightgbm_tuning_results.csv")
    if not tuning_path.exists():
        return {
            "n_estimators": 700,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "min_child_samples": 80,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
        }

    tuning_results = pd.read_csv(tuning_path).sort_values("selection_score")
    best_params = {
        key: tuning_results.iloc[0][key]
        for key in [
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
        ]
    }
    for integer_key in ["n_estimators", "num_leaves", "min_child_samples"]:
        best_params[integer_key] = int(best_params[integer_key])

    return best_params


def train_selected_quantiles(
    train_features: pd.DataFrame,
    train_prices: pd.Series,
    score_features: pd.DataFrame,
    quantile_params: Dict[str, object],
) -> pd.DataFrame:
    """Train Q10/Q50/Q90 and return dollar predictions for a scoring set."""

    quantile_levels = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
    predictions = pd.DataFrame(index=score_features.index)

    for quantile_name, quantile_level in quantile_levels.items():
        model = make_quantile_model(quantile_level, quantile_params)
        model.fit(train_features, np.log1p(train_prices))
        predictions[quantile_name] = np.expm1(model.predict(score_features))

    predictions[["q10", "q50", "q90"]] = np.sort(predictions[["q10", "q50", "q90"]].to_numpy(), axis=1)

    return predictions


def widen_intervals(predictions: pd.DataFrame, calibration_factor: float) -> pd.DataFrame:
    """Widen Q10 and Q90 around Q50 while preserving the median."""

    calibrated = predictions.copy()
    calibrated["q10"] = predictions["q50"] - calibration_factor * (predictions["q50"] - predictions["q10"])
    calibrated["q90"] = predictions["q50"] + calibration_factor * (predictions["q90"] - predictions["q50"])
    calibrated["q50"] = predictions["q50"]

    return calibrated


def interval_coverage(actual_prices: pd.Series, predictions: pd.DataFrame) -> float:
    """Calculate how often actual prices fall inside Q10-Q90."""

    return ((actual_prices >= predictions["q10"]) & (actual_prices <= predictions["q90"])).mean()


def choose_calibration_factor(
    validation_prices: pd.Series,
    validation_predictions: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Pick the smallest widening factor that reaches target validation coverage."""

    rows = []
    for factor in CALIBRATION_FACTORS:
        calibrated_predictions = widen_intervals(validation_predictions, factor)
        rows.append(
            {
                "calibration_factor": factor,
                "validation_coverage": interval_coverage(validation_prices, calibrated_predictions),
            }
        )

    factor_results = pd.DataFrame(rows)
    candidate_hits = factor_results[factor_results["validation_coverage"] >= TARGET_COVERAGE]

    if candidate_hits.empty:
        best_factor = factor_results.sort_values("validation_coverage").iloc[-1]["calibration_factor"]
    else:
        best_factor = candidate_hits.iloc[0]["calibration_factor"]

    return float(best_factor), factor_results


def summarize_interval_metrics(
    actual_prices: pd.Series,
    raw_predictions: pd.DataFrame,
    calibrated_predictions: pd.DataFrame,
) -> pd.Series:
    """Summarize raw vs calibrated test interval behavior."""

    return pd.Series(
        {
            "raw_q10_q90_coverage": interval_coverage(actual_prices, raw_predictions),
            "calibrated_q10_q90_coverage": interval_coverage(actual_prices, calibrated_predictions),
            "target_coverage": TARGET_COVERAGE,
            "raw_average_interval_width": (raw_predictions["q90"] - raw_predictions["q10"]).mean(),
            "calibrated_average_interval_width": (
                calibrated_predictions["q90"] - calibrated_predictions["q10"]
            ).mean(),
            "raw_average_uncertainty_score": (
                (raw_predictions["q90"] - raw_predictions["q10"]) / raw_predictions["q50"]
            ).mean(),
            "calibrated_average_uncertainty_score": (
                (calibrated_predictions["q90"] - calibrated_predictions["q10"]) / calibrated_predictions["q50"]
            ).mean(),
        }
    )


def plot_factor_search(factor_results: pd.DataFrame, chosen_factor: float) -> None:
    """Show validation coverage by calibration factor."""

    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.plot(
        factor_results["calibration_factor"],
        factor_results["validation_coverage"],
        marker="o",
        color="#4C78A8",
    )
    axis.axhline(TARGET_COVERAGE, linestyle="--", color="#E45756", label="Target 80%")
    axis.axvline(chosen_factor, linestyle="--", color="#F58518", label=f"Chosen factor {chosen_factor:.2f}")
    axis.set_title("Validation Coverage By Interval-Widening Factor")
    axis.set_xlabel("Calibration Factor")
    axis.set_ylabel("Q10-Q90 Coverage")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)

    fig.savefig(OUTPUT_DIR / "calibration_factor_search.png", dpi=160)
    plt.close(fig)


def plot_test_coverage_summary(metric_summary: pd.Series) -> None:
    """Compare raw and calibrated interval coverage on the test set."""

    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    labels = ["Target", "Raw", "Calibrated"]
    values = [
        metric_summary["target_coverage"],
        metric_summary["raw_q10_q90_coverage"],
        metric_summary["calibrated_q10_q90_coverage"],
    ]
    axis.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    axis.set_title("Test Interval Coverage Before And After Calibration")
    axis.set_ylabel("Q10-Q90 Coverage")
    axis.set_ylim(0, 1)

    for index, value in enumerate(values):
        axis.text(index, value + 0.03, f"{value:.1%}", ha="center")

    fig.savefig(OUTPUT_DIR / "test_coverage_before_after.png", dpi=160)
    plt.close(fig)


def plot_offer_shift(raw_offers: pd.DataFrame, calibrated_offers: pd.DataFrame) -> None:
    """Show how calibration changes offer-to-value ratios."""

    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.hist(
        raw_offers["buy_offer"] / raw_offers["median_price"],
        bins=40,
        alpha=0.60,
        label="Raw intervals",
        color="#F58518",
    )
    axis.hist(
        calibrated_offers["buy_offer"] / calibrated_offers["median_price"],
        bins=40,
        alpha=0.60,
        label="Calibrated intervals",
        color="#54A24B",
    )
    axis.set_title("Offer-To-Value Ratio Before And After Calibration")
    axis.set_xlabel("Buy Offer / Q50")
    axis.set_ylabel("Home Count")
    axis.legend(frameon=False)

    fig.savefig(OUTPUT_DIR / "offer_to_value_before_after.png", dpi=160)
    plt.close(fig)


def write_report(
    chosen_factor: float,
    metric_summary: pd.Series,
    raw_offers: pd.DataFrame,
    calibrated_offers: pd.DataFrame,
) -> None:
    """Write a Markdown report for the calibration step."""

    raw_offer_ratio = (raw_offers["buy_offer"] / raw_offers["median_price"]).mean()
    calibrated_offer_ratio = (calibrated_offers["buy_offer"] / calibrated_offers["median_price"]).mean()

    report = f"""# Quantile Interval Calibration

This step calibrates the raw `Q10-Q90` interval before using interval width in
the buy-offer engine.

## Why Calibration Is Needed

The quantile AVM's `Q50` is competitive as a fair-value estimate, but the raw
`Q10-Q90` interval is under-covered. That means the model is overconfident:
its stated 80% interval does not actually cover 80% of outcomes.

Calibration fixes the uncertainty estimate, not the median prediction.

## Method

The validation period is used to choose a global interval-widening factor:

```text
Q10_calibrated = Q50 - factor * (Q50 - Q10_raw)
Q90_calibrated = Q50 + factor * (Q90_raw - Q50)
Q50_calibrated = Q50
```

The chosen factor is the smallest factor that reaches 80% validation coverage.

Chosen calibration factor:

```text
{chosen_factor:.2f}
```

![Calibration factor search](calibration_factor_search.png)

## Test Set Results

{metric_summary.to_frame("value").to_markdown(floatfmt=".4f")}

![Coverage before and after](test_coverage_before_after.png)

## Impact On Buy Offers

Calibration widens uncertainty bands, which increases the uncertainty penalty
and makes offers more conservative.

```text
Average raw offer / Q50:        {raw_offer_ratio:.2%}
Average calibrated offer / Q50: {calibrated_offer_ratio:.2%}
```

![Offer shift](offer_to_value_before_after.png)

## Interpretation

The calibrated intervals are more reliable for risk-aware pricing. This is a
reliability layer: it does not replace the model, and it does not change the
fair-value estimate. It makes the model's uncertainty estimates more honest
before they are converted into acquisition spreads.

"""

    (OUTPUT_DIR / "quantile_calibration_report.md").write_text(report)


def main() -> None:
    """Run validation-based interval calibration and test evaluation."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_features, test_features, train_prices, test_prices = load_modeling_data()
    quantile_params = load_best_quantile_params()

    model_features, validation_features, model_prices, validation_prices = train_test_split(
        train_features,
        train_prices,
        test_size=0.20,
        shuffle=False,
    )

    validation_predictions = train_selected_quantiles(
        model_features,
        model_prices,
        validation_features,
        quantile_params,
    )
    chosen_factor, factor_results = choose_calibration_factor(validation_prices, validation_predictions)

    raw_test_predictions = train_selected_quantiles(
        train_features,
        train_prices,
        test_features,
        quantile_params,
    )
    calibrated_test_predictions = widen_intervals(raw_test_predictions, chosen_factor)

    metric_summary = summarize_interval_metrics(test_prices, raw_test_predictions, calibrated_test_predictions)

    spread_assumptions = SpreadAssumptions()
    raw_offers = calculate_buy_offer(
        raw_test_predictions["q50"],
        raw_test_predictions["q10"],
        raw_test_predictions["q90"],
        spread_assumptions,
    )
    calibrated_offers = calculate_buy_offer(
        calibrated_test_predictions["q50"],
        calibrated_test_predictions["q10"],
        calibrated_test_predictions["q90"],
        spread_assumptions,
    )

    factor_results.to_csv(OUTPUT_DIR / "calibration_factor_search.csv", index=False)
    metric_summary.to_csv(OUTPUT_DIR / "calibration_summary_metrics.csv", header=["value"])
    raw_test_predictions.to_csv(OUTPUT_DIR / "raw_test_quantiles.csv", index=False)
    calibrated_test_predictions.to_csv(OUTPUT_DIR / "calibrated_test_quantiles.csv", index=False)

    plot_factor_search(factor_results, chosen_factor)
    plot_test_coverage_summary(metric_summary)
    plot_offer_shift(raw_offers, calibrated_offers)
    write_report(chosen_factor, metric_summary, raw_offers, calibrated_offers)

    print("\nCalibration factor:", round(chosen_factor, 4))
    print("\nCalibration summary:")
    print(metric_summary.round(4))
    print(f"\nSaved calibration report to {OUTPUT_DIR / 'quantile_calibration_report.md'}")


if __name__ == "__main__":
    main()
