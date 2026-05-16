"""Vanilla point-prediction AVM baseline.

This script compares a standard LightGBM regression AVM against the Q50 median
prediction from the probabilistic quantile AVM. It uses the exact same King
County + FRED data pipeline so the comparison is fair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from risk_adjusted_pricing import RANDOM_STATE, load_modeling_data, make_quantile_model


OUTPUT_DIR = Path("outputs/2_point_baseline")


def mean_absolute_percentage_error(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate average absolute percentage error."""

    return ((actual - predicted).abs() / actual).mean()


def median_absolute_percentage_error(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate median absolute percentage error."""

    return ((actual - predicted).abs() / actual).median()


def tune_point_model_parameters(
    train_features: pd.DataFrame,
    train_prices: pd.Series,
) -> tuple[Dict[str, object], pd.DataFrame]:
    """Tune a small standard LightGBM regression grid on validation MAPE.

    Why: this baseline should be a fair point-prediction AVM, not an untuned
    strawman. It uses the same time-aware validation style as the quantile AVM.
    """

    log_train_prices = np.log1p(train_prices)

    model_features, validation_features, model_prices, validation_prices = train_test_split(
        train_features,
        log_train_prices,
        test_size=0.20,
        shuffle=False,
    )

    candidate_parameters = [
        {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 80, "subsample": 0.90, "colsample_bytree": 0.90},
        {"n_estimators": 700, "learning_rate": 0.035, "num_leaves": 31, "min_child_samples": 80, "subsample": 0.90, "colsample_bytree": 0.90},
        {"n_estimators": 700, "learning_rate": 0.035, "num_leaves": 63, "min_child_samples": 200, "subsample": 0.85, "colsample_bytree": 0.80},
    ]

    tuning_rows = []
    for candidate_index, candidate_params in enumerate(candidate_parameters, start=1):
        model = LGBMRegressor(
            objective="regression",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            **candidate_params,
        )
        model.fit(model_features, model_prices)

        validation_predictions = np.expm1(model.predict(validation_features))
        validation_actuals = np.expm1(validation_prices)

        tuning_rows.append(
            {
                "candidate": candidate_index,
                "validation_mae": mean_absolute_error(validation_actuals, validation_predictions),
                "validation_mape": mean_absolute_percentage_error(validation_actuals, validation_predictions),
                "validation_rmse": root_mean_squared_error(validation_actuals, validation_predictions),
                **candidate_params,
            }
        )

    tuning_results = pd.DataFrame(tuning_rows).sort_values("validation_mape")
    best_params = {key: tuning_results.iloc[0][key] for key in candidate_parameters[0].keys()}

    for integer_key in ["n_estimators", "num_leaves", "min_child_samples"]:
        best_params[integer_key] = int(best_params[integer_key])

    return best_params, tuning_results


def train_point_avm(
    train_features: pd.DataFrame,
    train_prices: pd.Series,
    test_features: pd.DataFrame,
) -> tuple[pd.Series, Dict[str, object], pd.DataFrame]:
    """Train the vanilla point AVM and return test predictions."""

    best_params, tuning_results = tune_point_model_parameters(train_features, train_prices)
    model = LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        **best_params,
    )
    model.fit(train_features, np.log1p(train_prices))

    predictions = pd.Series(
        np.expm1(model.predict(test_features)),
        index=test_features.index,
        name="point_avm_prediction",
    )

    return predictions, best_params, tuning_results


def load_best_quantile_params() -> Dict[str, object]:
    """Reuse the best quantile parameters from the main AVM run when available."""

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
        for key in ["n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree"]
    }
    for integer_key in ["n_estimators", "num_leaves", "min_child_samples"]:
        best_params[integer_key] = int(best_params[integer_key])

    return best_params


def train_quantile_q50_avm(
    train_features: pd.DataFrame,
    train_prices: pd.Series,
    test_features: pd.DataFrame,
) -> tuple[pd.Series, Dict[str, object]]:
    """Train only the Q50 quantile model needed for the point comparison."""

    quantile_params = load_best_quantile_params()
    model = make_quantile_model(0.50, quantile_params)
    model.fit(train_features, np.log1p(train_prices))

    predictions = pd.Series(
        np.expm1(model.predict(test_features)),
        index=test_features.index,
        name="quantile_q50_prediction",
    )

    return predictions, quantile_params


def score_predictions(actual_prices: pd.Series, predictions: pd.Series) -> pd.Series:
    """Return point-prediction metrics."""

    return pd.Series(
        {
            "mae": mean_absolute_error(actual_prices, predictions),
            "mape": mean_absolute_percentage_error(actual_prices, predictions),
            "median_ape": median_absolute_percentage_error(actual_prices, predictions),
            "rmse": root_mean_squared_error(actual_prices, predictions),
        }
    )


def plot_metric_comparison(metric_table: pd.DataFrame) -> None:
    """Save a compact metric comparison plot."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    metric_table.loc[["Vanilla Point AVM", "Quantile AVM Q50"], "mape"].plot(
        kind="bar",
        ax=axes[0],
        color=["#4C78A8", "#F58518"],
    )
    axes[0].set_title("MAPE Comparison")
    axes[0].set_ylabel("MAPE")
    axes[0].tick_params(axis="x", rotation=15)

    metric_table.loc[["Vanilla Point AVM", "Quantile AVM Q50"], "mae"].plot(
        kind="bar",
        ax=axes[1],
        color=["#4C78A8", "#F58518"],
    )
    axes[1].set_title("MAE Comparison")
    axes[1].set_ylabel("MAE ($)")
    axes[1].tick_params(axis="x", rotation=15)

    fig.savefig(OUTPUT_DIR / "point_vs_q50_metric_comparison.png", dpi=160)
    plt.close(fig)


def plot_prediction_scatter(comparison_frame: pd.DataFrame) -> None:
    """Save actual-vs-predicted scatter plots for both models."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    max_price = comparison_frame["actual_price"].quantile(0.99)

    for axis, prediction_column, title, color in [
        (axes[0], "point_avm_prediction", "Vanilla Point AVM", "#4C78A8"),
        (axes[1], "quantile_q50_prediction", "Quantile AVM Q50", "#F58518"),
    ]:
        axis.scatter(
            comparison_frame["actual_price"],
            comparison_frame[prediction_column],
            s=10,
            alpha=0.25,
            color=color,
        )
        axis.plot([0, max_price], [0, max_price], linestyle="--", color="black", linewidth=1)
        axis.set_xlim(0, max_price)
        axis.set_ylim(0, max_price)
        axis.set_title(title)
        axis.set_xlabel("Actual Price ($)")
        axis.set_ylabel("Predicted Price ($)")

    fig.savefig(OUTPUT_DIR / "actual_vs_predicted_comparison.png", dpi=160)
    plt.close(fig)


def write_report(
    metric_table: pd.DataFrame,
    point_params: Dict[str, object],
    quantile_params: Dict[str, object],
) -> None:
    """Write a Markdown report comparing point AVM vs Q50."""

    point_mape = metric_table.loc["Vanilla Point AVM", "mape"]
    q50_mape = metric_table.loc["Quantile AVM Q50", "mape"]
    mape_delta = point_mape - q50_mape

    if abs(mape_delta) < 0.0025:
        takeaway = "The two models are very close on normalized point accuracy."
    elif mape_delta < 0:
        takeaway = "The vanilla point AVM has better normalized point accuracy."
    else:
        takeaway = "The quantile AVM Q50 has better normalized point accuracy."

    report = f"""# Vanilla Point AVM vs Quantile AVM Q50

This comparison checks whether the probabilistic AVM gives up too much point
prediction accuracy compared with a standard regression AVM.

Both models use the same King County + FRED feature pipeline and the same
time-based train/test split.

## Why This Comparison Matters

The quantile AVM is valuable because it produces uncertainty bands. But if its
median prediction is much worse than a standard point model, a production design
might use:

```text
point model -> fair value
quantile model -> uncertainty band
offer engine -> combines both
```

If Q50 is competitive, the simpler probabilistic architecture is easier to
defend.

## Metric Comparison

{metric_table.to_markdown(floatfmt=".4f")}

![Metric comparison](point_vs_q50_metric_comparison.png)

![Actual vs predicted comparison](actual_vs_predicted_comparison.png)

## Takeaway

{takeaway}

MAPE difference, defined as `point_mape - q50_mape`, is `{mape_delta:.4f}`.
A negative value means the vanilla point AVM is better; a positive value means
the quantile Q50 is better.

## Best Point AVM Parameters

```text
{pd.Series(point_params).to_string()}
```

## Best Quantile AVM Parameters

```text
{pd.Series(quantile_params).to_string()}
```

"""

    (OUTPUT_DIR / "point_vs_q50_report.md").write_text(report)


def main() -> None:
    """Run the point AVM baseline and compare it to quantile Q50."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_features, test_features, train_prices, test_prices = load_modeling_data()

    point_predictions, point_params, point_tuning_results = train_point_avm(
        train_features,
        train_prices,
        test_features,
    )

    q50_predictions, quantile_params = train_quantile_q50_avm(
        train_features,
        train_prices,
        test_features,
    )

    comparison_frame = pd.DataFrame(
        {
            "actual_price": test_prices,
            "point_avm_prediction": point_predictions,
            "quantile_q50_prediction": q50_predictions,
        }
    )
    comparison_frame["point_absolute_percentage_error"] = (
        comparison_frame["actual_price"] - comparison_frame["point_avm_prediction"]
    ).abs() / comparison_frame["actual_price"]
    comparison_frame["q50_absolute_percentage_error"] = (
        comparison_frame["actual_price"] - comparison_frame["quantile_q50_prediction"]
    ).abs() / comparison_frame["actual_price"]

    metric_table = pd.DataFrame(
        {
            "Vanilla Point AVM": score_predictions(test_prices, point_predictions),
            "Quantile AVM Q50": score_predictions(test_prices, q50_predictions),
        }
    ).T

    metric_table.to_csv(OUTPUT_DIR / "point_vs_q50_metrics.csv")
    comparison_frame.to_csv(OUTPUT_DIR / "point_vs_q50_predictions.csv", index=False)
    point_tuning_results.to_csv(OUTPUT_DIR / "point_avm_tuning_results.csv", index=False)

    plot_metric_comparison(metric_table)
    plot_prediction_scatter(comparison_frame)
    write_report(metric_table, point_params, quantile_params)

    print("\nPoint AVM vs Quantile Q50 metrics:")
    print(metric_table.round(4))
    print(f"\nSaved comparison report to {OUTPUT_DIR / 'point_vs_q50_report.md'}")


if __name__ == "__main__":
    main()
