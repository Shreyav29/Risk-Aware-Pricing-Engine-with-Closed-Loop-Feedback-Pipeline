"""Compare width-scaling calibration against isotonic quantile calibration.

Both calibration methods are learned on the validation period and evaluated on
the untouched test period. This answers: which calibration layer should we use
before converting uncertainty bands into buy-offer spreads?
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from risk_adjusted_pricing import (
    SpreadAssumptions,
    calculate_buy_offer,
    load_modeling_data,
    make_quantile_model,
)


OUTPUT_DIR = Path("outputs/5_calibration_method_comparison")
TARGET_COVERAGE = 0.80
MODEL_QUANTILE_LEVELS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
REPORT_QUANTILE_LEVELS = np.arange(0.10, 1.00, 0.10)
WIDTH_FACTORS = np.arange(1.00, 2.55, 0.05)


def quantile_column(quantile_level: float) -> str:
    """Return a stable column name for a quantile level."""

    return f"q{int(round(quantile_level * 100)):02d}"


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


def train_quantile_grid(
    train_features: pd.DataFrame,
    train_prices: pd.Series,
    score_features: pd.DataFrame,
    quantile_levels: np.ndarray,
    quantile_params: Dict[str, object],
) -> pd.DataFrame:
    """Train a grid of quantile models and return sorted dollar predictions."""

    predictions = pd.DataFrame(index=score_features.index)

    for quantile_level in quantile_levels:
        column = quantile_column(quantile_level)
        model = make_quantile_model(float(quantile_level), quantile_params)
        model.fit(train_features, np.log1p(train_prices))
        predictions[column] = np.expm1(model.predict(score_features))

    quantile_columns = [quantile_column(level) for level in quantile_levels]
    predictions[quantile_columns] = np.sort(predictions[quantile_columns].to_numpy(), axis=1)

    return predictions


def interpolate_quantiles(
    raw_predictions: pd.DataFrame,
    raw_levels: np.ndarray,
    target_levels: np.ndarray,
) -> pd.DataFrame:
    """Interpolate per-home quantile predictions at requested levels."""

    raw_columns = [quantile_column(level) for level in raw_levels]
    interpolated = pd.DataFrame(index=raw_predictions.index)

    raw_values = raw_predictions[raw_columns].to_numpy()
    for target_level in target_levels:
        interpolated[quantile_column(target_level)] = [
            np.interp(target_level, raw_levels, row_values) for row_values in raw_values
        ]

    return interpolated


def widen_all_quantiles(
    predictions: pd.DataFrame,
    calibration_factor: float,
    quantile_levels: np.ndarray,
) -> pd.DataFrame:
    """Apply width scaling to every reported quantile around Q50."""

    calibrated = pd.DataFrame(index=predictions.index)
    q50 = predictions["q50"]

    for quantile_level in quantile_levels:
        column = quantile_column(quantile_level)
        if quantile_level < 0.50:
            calibrated[column] = q50 - calibration_factor * (q50 - predictions[column])
        elif quantile_level > 0.50:
            calibrated[column] = q50 + calibration_factor * (predictions[column] - q50)
        else:
            calibrated[column] = q50

    return calibrated


def interval_coverage(actual_prices: pd.Series, predictions: pd.DataFrame) -> float:
    """Calculate Q10-Q90 interval coverage."""

    return ((actual_prices >= predictions["q10"]) & (actual_prices <= predictions["q90"])).mean()


def choose_width_factor(validation_prices: pd.Series, validation_predictions: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Pick the smallest width factor that reaches 80% validation coverage."""

    rows = []
    for factor in WIDTH_FACTORS:
        calibrated = widen_all_quantiles(validation_predictions, factor, REPORT_QUANTILE_LEVELS)
        rows.append(
            {
                "calibration_factor": factor,
                "validation_coverage": interval_coverage(validation_prices, calibrated),
            }
        )

    factor_results = pd.DataFrame(rows)
    hits = factor_results[factor_results["validation_coverage"] >= TARGET_COVERAGE]
    chosen_factor = hits.iloc[0]["calibration_factor"] if not hits.empty else factor_results.iloc[-1]["calibration_factor"]

    return float(chosen_factor), factor_results


def observed_quantile_frequencies(
    actual_prices: pd.Series,
    predictions: pd.DataFrame,
    quantile_levels: np.ndarray,
) -> pd.DataFrame:
    """Calculate observed frequency below each predicted quantile."""

    rows = []
    for quantile_level in quantile_levels:
        column = quantile_column(quantile_level)
        rows.append(
            {
                "predicted_quantile": quantile_level,
                "observed_frequency": (actual_prices <= predictions[column]).mean(),
                "calibration_error": (actual_prices <= predictions[column]).mean() - quantile_level,
            }
        )

    return pd.DataFrame(rows)


def fit_isotonic_quantile_mapper(
    validation_prices: pd.Series,
    validation_predictions: pd.DataFrame,
    raw_levels: np.ndarray,
) -> tuple[IsotonicRegression, pd.DataFrame]:
    """Fit isotonic mapping from raw quantile level to observed frequency."""

    reliability = observed_quantile_frequencies(validation_prices, validation_predictions, raw_levels)
    isotonic_model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    isotonic_model.fit(reliability["predicted_quantile"], reliability["observed_frequency"])

    reliability["isotonic_observed_frequency"] = isotonic_model.predict(reliability["predicted_quantile"])

    return isotonic_model, reliability


def invert_isotonic_mapping(
    isotonic_model: IsotonicRegression,
    target_levels: np.ndarray,
) -> np.ndarray:
    """Map target calibrated levels to raw quantile levels via inverse interpolation."""

    raw_grid = np.linspace(MODEL_QUANTILE_LEVELS.min(), MODEL_QUANTILE_LEVELS.max(), 500)
    observed_grid = isotonic_model.predict(raw_grid)

    # Repeated observed values can happen with isotonic plateaus. Grouping keeps
    # the inverse interpolation monotonic and stable.
    inverse_frame = pd.DataFrame({"raw_level": raw_grid, "observed_frequency": observed_grid})
    inverse_frame = inverse_frame.groupby("observed_frequency", as_index=False)["raw_level"].mean()

    adjusted_raw_levels = np.interp(
        target_levels,
        inverse_frame["observed_frequency"],
        inverse_frame["raw_level"],
        left=inverse_frame["raw_level"].min(),
        right=inverse_frame["raw_level"].max(),
    )

    return adjusted_raw_levels


def apply_isotonic_calibration(
    raw_predictions: pd.DataFrame,
    isotonic_model: IsotonicRegression,
    target_levels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create calibrated quantile predictions using the isotonic inverse map."""

    adjusted_raw_levels = invert_isotonic_mapping(isotonic_model, target_levels)
    raw_columns = [quantile_column(level) for level in MODEL_QUANTILE_LEVELS]
    raw_values = raw_predictions[raw_columns].to_numpy()
    calibrated = pd.DataFrame(index=raw_predictions.index)

    for target_level, adjusted_raw_level in zip(target_levels, adjusted_raw_levels):
        calibrated[quantile_column(target_level)] = [
            np.interp(adjusted_raw_level, MODEL_QUANTILE_LEVELS, row_values) for row_values in raw_values
        ]

    mapping = pd.DataFrame(
        {
            "target_quantile": target_levels,
            "adjusted_raw_quantile_level": adjusted_raw_levels,
        }
    )

    return calibrated, mapping


def score_calibration_method(
    method_name: str,
    actual_prices: pd.Series,
    predictions: pd.DataFrame,
) -> pd.Series:
    """Compute coverage, sharpness, reliability, and offer metrics."""

    reliability = observed_quantile_frequencies(actual_prices, predictions, REPORT_QUANTILE_LEVELS)
    spread_assumptions = SpreadAssumptions()
    offers = calculate_buy_offer(predictions["q50"], predictions["q10"], predictions["q90"], spread_assumptions)

    return pd.Series(
        {
            "method": method_name,
            "q10_q90_coverage": interval_coverage(actual_prices, predictions),
            "target_coverage": TARGET_COVERAGE,
            "mean_abs_quantile_calibration_error": reliability["calibration_error"].abs().mean(),
            "average_interval_width": (predictions["q90"] - predictions["q10"]).mean(),
            "average_uncertainty_score": ((predictions["q90"] - predictions["q10"]) / predictions["q50"]).mean(),
            "average_offer_to_q50": (offers["buy_offer"] / offers["median_price"]).mean(),
        }
    )


def plot_coverage_and_width(comparison_table: pd.DataFrame) -> None:
    """Compare coverage and interval width across methods."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axes[0].bar(comparison_table.index, comparison_table["q10_q90_coverage"], color=["#F58518", "#54A24B", "#4C78A8"])
    axes[0].axhline(TARGET_COVERAGE, linestyle="--", color="black", label="Target 80%")
    axes[0].set_title("Test Q10-Q90 Coverage")
    axes[0].set_ylabel("Coverage")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(frameon=False)

    for index, value in enumerate(comparison_table["q10_q90_coverage"]):
        axes[0].text(index, value + 0.03, f"{value:.1%}", ha="center")

    axes[1].bar(comparison_table.index, comparison_table["average_interval_width"], color=["#F58518", "#54A24B", "#4C78A8"])
    axes[1].set_title("Average Q10-Q90 Width")
    axes[1].set_ylabel("Dollars")
    axes[1].tick_params(axis="x", rotation=15)

    fig.savefig(OUTPUT_DIR / "calibration_method_coverage_width.png", dpi=160)
    plt.close(fig)


def plot_reliability_curves(reliability_tables: dict[str, pd.DataFrame]) -> None:
    """Plot predicted quantile vs observed frequency for all methods."""

    fig, axis = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")

    colors = {"Raw": "#F58518", "Width Scaling": "#54A24B", "Isotonic": "#4C78A8"}
    for method_name, reliability in reliability_tables.items():
        axis.plot(
            reliability["predicted_quantile"],
            reliability["observed_frequency"],
            marker="o",
            color=colors[method_name],
            label=method_name,
        )

    axis.set_title("Quantile Reliability By Calibration Method")
    axis.set_xlabel("Predicted Quantile")
    axis.set_ylabel("Observed Frequency")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    fig.savefig(OUTPUT_DIR / "calibration_method_reliability_curves.png", dpi=160)
    plt.close(fig)


def write_report(
    comparison_table: pd.DataFrame,
    chosen_width_factor: float,
    isotonic_mapping: pd.DataFrame,
) -> None:
    """Write a Markdown comparison report."""

    best_coverage_method = (comparison_table["q10_q90_coverage"] - TARGET_COVERAGE).abs().idxmin()
    best_reliability_method = comparison_table["mean_abs_quantile_calibration_error"].idxmin()
    sharpest_near_target = comparison_table[comparison_table["q10_q90_coverage"] >= 0.75]["average_interval_width"].idxmin()

    report = f"""# Calibration Method Comparison

This report compares two calibration approaches:

1. **Width scaling**: widen `Q10` and `Q90` around `Q50` using one validation-selected factor.
2. **Isotonic quantile calibration**: use the validation reliability curve to map desired calibrated quantiles to adjusted raw quantile levels.

All calibration choices are learned on the validation period and evaluated on the untouched test period.

## Method Details

Width scaling selected factor:

```text
{chosen_width_factor:.2f}
```

Isotonic target-to-raw quantile mapping:

{isotonic_mapping.to_markdown(index=False, floatfmt=".3f")}

## Test Metrics

{comparison_table.to_markdown(floatfmt=".4f")}

![Coverage and width](calibration_method_coverage_width.png)

![Reliability curves](calibration_method_reliability_curves.png)

## Interpretation

- Best method by closeness to 80% interval coverage: **{best_coverage_method}**
- Best method by mean absolute quantile calibration error: **{best_reliability_method}**
- Sharpest method among methods with at least 75% coverage: **{sharpest_near_target}**

Width scaling is more transparent and preserves the original `Q50` fair-value estimate.
Isotonic calibration is more flexible and can correct nonlinear reliability-curve distortions, but it can also shift the effective quantile levels, including the median.

For this pricing system, the preferred method should balance:

```text
coverage close to 80%
+ low quantile calibration error
+ reasonably narrow intervals
+ explainability for business stakeholders
```

"""

    (OUTPUT_DIR / "calibration_method_comparison_report.md").write_text(report)


def main() -> None:
    """Run width-scaling vs isotonic calibration comparison."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_features, test_features, train_prices, test_prices = load_modeling_data()
    quantile_params = load_best_quantile_params()

    model_features, validation_features, model_prices, validation_prices = train_test_split(
        train_features,
        train_prices,
        test_size=0.20,
        shuffle=False,
    )

    validation_grid = train_quantile_grid(
        model_features,
        model_prices,
        validation_features,
        MODEL_QUANTILE_LEVELS,
        quantile_params,
    )
    raw_validation_deciles = interpolate_quantiles(validation_grid, MODEL_QUANTILE_LEVELS, REPORT_QUANTILE_LEVELS)

    chosen_width_factor, width_factor_results = choose_width_factor(validation_prices, raw_validation_deciles)

    isotonic_model, validation_isotonic_reliability = fit_isotonic_quantile_mapper(
        validation_prices,
        validation_grid,
        MODEL_QUANTILE_LEVELS,
    )

    test_grid = train_quantile_grid(
        train_features,
        train_prices,
        test_features,
        MODEL_QUANTILE_LEVELS,
        quantile_params,
    )
    raw_test_deciles = interpolate_quantiles(test_grid, MODEL_QUANTILE_LEVELS, REPORT_QUANTILE_LEVELS)
    width_scaled_test_deciles = widen_all_quantiles(raw_test_deciles, chosen_width_factor, REPORT_QUANTILE_LEVELS)
    isotonic_test_deciles, isotonic_mapping = apply_isotonic_calibration(
        test_grid,
        isotonic_model,
        REPORT_QUANTILE_LEVELS,
    )

    predictions_by_method = {
        "Raw": raw_test_deciles,
        "Width Scaling": width_scaled_test_deciles,
        "Isotonic": isotonic_test_deciles,
    }

    comparison_table = pd.DataFrame(
        [score_calibration_method(method, test_prices, predictions) for method, predictions in predictions_by_method.items()]
    ).set_index("method")

    reliability_tables = {
        method: observed_quantile_frequencies(test_prices, predictions, REPORT_QUANTILE_LEVELS)
        for method, predictions in predictions_by_method.items()
    }

    comparison_table.to_csv(OUTPUT_DIR / "calibration_method_metrics.csv")
    width_factor_results.to_csv(OUTPUT_DIR / "width_factor_search.csv", index=False)
    validation_isotonic_reliability.to_csv(OUTPUT_DIR / "validation_isotonic_reliability.csv", index=False)
    isotonic_mapping.to_csv(OUTPUT_DIR / "isotonic_quantile_mapping.csv", index=False)
    for method, reliability in reliability_tables.items():
        reliability.to_csv(OUTPUT_DIR / f"{method.lower().replace(' ', '_')}_test_reliability.csv", index=False)

    plot_coverage_and_width(comparison_table)
    plot_reliability_curves(reliability_tables)
    write_report(comparison_table, chosen_width_factor, isotonic_mapping)

    print("\nCalibration method comparison:")
    print(comparison_table.round(4))
    print(f"\nSaved report to {OUTPUT_DIR / 'calibration_method_comparison_report.md'}")


if __name__ == "__main__":
    main()
