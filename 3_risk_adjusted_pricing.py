"""Risk-aware home pricing engine for real estate acquisition decisions.

This version uses the King County home sales dataset and enriches each sale
with macroeconomic signals from FRED. The code stays readable and modular on
purpose: the goal is to make the pricing logic easy to audit, explain, and extend.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split


RANDOM_STATE = 42
DATA_PATH = "kc_house_data.csv"
OUTPUT_DIR = Path("outputs")
DATA_OUTPUT_DIR = OUTPUT_DIR / "3_data"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "3_model_diagnostics"
OFFER_OUTPUT_DIR = OUTPUT_DIR / "3_offer_engine"
FRED_CACHE_PATH = DATA_OUTPUT_DIR / "fred_macro_features.csv"
CALIBRATION_TARGET_COVERAGE = 0.80
CALIBRATION_FACTOR_GRID = np.arange(1.00, 2.55, 0.05)

FRED_SERIES = {
    "MORTGAGE30US": "mortgage_30y_rate",
    "SEXRNSA": "case_shiller_seattle",
    "SEAT653UR": "seattle_unemployment_rate",
    "DGS10": "treasury_10y_rate",
    "UMCSENT": "consumer_sentiment",
    "MSACSR": "months_supply_homes",
    "HOUST": "housing_starts",
    "CPIAUCSL": "consumer_price_index",
}


@dataclass
class SpreadAssumptions:
    """Business assumptions that convert fair value into a buy offer."""

    transaction_cost_rate: float = 0.015
    holding_cost_rate: float = 0.010
    resale_prep_cost_rate: float = 0.015
    target_margin_rate: float = 0.010
    penalty_strength: float = 0.15
    max_uncertainty_penalty: float = 0.10

    @property
    def base_spread(self) -> float:
        """Minimum spread needed even when valuation uncertainty is low."""
        return (
            self.transaction_cost_rate
            + self.holding_cost_rate
            + self.resale_prep_cost_rate
            + self.target_margin_rate
        )


def load_dotenv_value(key: str, env_path: str = ".env") -> str | None:
    """Read a simple KEY=value pair from the environment or a local .env file."""

    if os.environ.get(key):
        return os.environ[key]

    path = Path(env_path)
    if not path.exists():
        return None

    for line in path.read_text().splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")

    return None


def fetch_fred_series(
    series_id: str,
    api_key: str,
    observation_start: pd.Timestamp,
    observation_end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch one FRED series and return date/value observations."""

    query = urllib.parse.urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start.strftime("%Y-%m-%d"),
            "observation_end": observation_end.strftime("%Y-%m-%d"),
        }
    )
    url = f"https://api.stlouisfed.org/fred/series/observations?{query}"

    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    observations = pd.DataFrame(payload["observations"])
    observations["date"] = pd.to_datetime(observations["date"])
    observations["value"] = pd.to_numeric(observations["value"].replace(".", np.nan), errors="coerce")

    return observations[["date", "value"]]


def build_macro_features(
    sale_dates: pd.Series,
    fred_series: Dict[str, str] = FRED_SERIES,
    cache_path: Path = FRED_CACHE_PATH,
) -> pd.DataFrame:
    """Fetch FRED macro variables and convert them to monthly features."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["sale_month"])

    legacy_cache_path = OUTPUT_DIR / "fred_macro_features.csv"
    if legacy_cache_path.exists():
        macro_frame = pd.read_csv(legacy_cache_path, parse_dates=["sale_month"])
        macro_frame.to_csv(cache_path, index=False)
        return macro_frame

    api_key = load_dotenv_value("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY was not found in the environment or .env file.")

    observation_start = sale_dates.min() - pd.DateOffset(months=6)
    observation_end = sale_dates.max() + pd.DateOffset(months=1)

    monthly_series = []
    for series_id, feature_name in fred_series.items():
        series_frame = fetch_fred_series(series_id, api_key, observation_start, observation_end)

        # Why: daily and weekly macro series are noisy at the transaction level.
        # Monthly averages match the sale-date granularity used in this mock AVM.
        monthly_frame = (
            series_frame.set_index("date")["value"]
            .resample("MS")
            .mean()
            .rename(feature_name)
            .to_frame()
        )
        monthly_series.append(monthly_frame)

    macro_frame = pd.concat(monthly_series, axis=1).sort_index()
    macro_frame = macro_frame.ffill().bfill()
    macro_frame.index.name = "sale_month"
    macro_frame = macro_frame.reset_index()

    # Why: levels matter, but month-over-month changes help the model see market
    # direction over the short King County sample.
    for feature_name in fred_series.values():
        macro_frame[f"{feature_name}_mom_change"] = macro_frame[feature_name].pct_change().fillna(0)

    macro_frame.to_csv(cache_path, index=False)
    return macro_frame


def parse_king_county_sales(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load King County sales, parse dates, and attach monthly macro features."""

    sales = pd.read_csv(data_path)
    sales["sale_date"] = pd.to_datetime(sales["date"], format="%Y%m%dT%H%M%S")
    sales["sale_month"] = sales["sale_date"].values.astype("datetime64[M]")

    macro_features = build_macro_features(sales["sale_date"])
    sales = sales.merge(macro_features, on="sale_month", how="left")
    sales = sales.sort_values("sale_date").reset_index(drop=True)

    return sales


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two columns while avoiding infinite values from zero denominators."""

    return numerator / denominator.replace(0, np.nan)


def add_king_county_features(
    train_features_raw: pd.DataFrame,
    test_features_raw: pd.DataFrame,
    train_prices: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create King County AVM features without leaking test outcomes."""

    train_features = train_features_raw.copy()
    test_features = test_features_raw.copy()

    for feature_frame in [train_features, test_features]:
        feature_frame.drop(columns=["id", "date"], inplace=True, errors="ignore")

        feature_frame["sale_year"] = feature_frame["sale_date"].dt.year
        feature_frame["sale_month_number"] = feature_frame["sale_date"].dt.month
        feature_frame["sale_quarter"] = feature_frame["sale_date"].dt.quarter
        feature_frame["is_spring_summer_sale"] = feature_frame["sale_month_number"].isin([4, 5, 6, 7, 8]).astype(int)

        feature_frame["home_age"] = feature_frame["sale_year"] - feature_frame["yr_built"]
        feature_frame["was_renovated"] = (feature_frame["yr_renovated"] > 0).astype(int)
        feature_frame["years_since_renovation"] = np.where(
            feature_frame["yr_renovated"] > 0,
            feature_frame["sale_year"] - feature_frame["yr_renovated"],
            feature_frame["home_age"],
        )

        feature_frame["has_basement"] = (feature_frame["sqft_basement"] > 0).astype(int)
        feature_frame["basement_share"] = _safe_divide(feature_frame["sqft_basement"], feature_frame["sqft_living"]).fillna(0)
        feature_frame["lot_to_living_ratio"] = _safe_divide(feature_frame["sqft_lot"], feature_frame["sqft_living"]).fillna(0)
        feature_frame["nearby_living_ratio"] = _safe_divide(feature_frame["sqft_living"], feature_frame["sqft_living15"]).fillna(0)
        feature_frame["nearby_lot_ratio"] = _safe_divide(feature_frame["sqft_lot"], feature_frame["sqft_lot15"]).fillna(0)
        feature_frame["bathrooms_per_bedroom"] = _safe_divide(feature_frame["bathrooms"], feature_frame["bedrooms"]).fillna(0)
        feature_frame["living_area_per_bedroom"] = _safe_divide(feature_frame["sqft_living"], feature_frame["bedrooms"]).fillna(0)

        feature_frame["grade_x_living_area"] = feature_frame["grade"] * feature_frame["sqft_living"]
        feature_frame["condition_x_living_area"] = feature_frame["condition"] * feature_frame["sqft_living"]
        feature_frame["view_x_waterfront"] = feature_frame["view"] * feature_frame["waterfront"]
        feature_frame["grade_x_condition"] = feature_frame["grade"] * feature_frame["condition"]

        # Why: Seattle-area location has strong nonlinear effects. Distance-like
        # features give the model smooth geographic signals beyond zipcode.
        feature_frame["distance_to_seattle_center"] = np.sqrt(
            (feature_frame["lat"] - 47.6062) ** 2 + (feature_frame["long"] + 122.3321) ** 2
        )
        feature_frame["lat_long_interaction"] = feature_frame["lat"] * feature_frame["long"]

        for skewed_column in [
            "sqft_living",
            "sqft_lot",
            "sqft_above",
            "sqft_basement",
            "sqft_living15",
            "sqft_lot15",
        ]:
            feature_frame[f"log_{skewed_column}"] = np.log1p(feature_frame[skewed_column])

    add_zipcode_price_prior(train_features, test_features, train_prices)

    for feature_frame in [train_features, test_features]:
        feature_frame.drop(columns=["sale_date", "sale_month"], inplace=True, errors="ignore")
        feature_frame["zipcode"] = feature_frame["zipcode"].astype(str)

    return train_features, test_features


def add_zipcode_price_prior(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_prices: pd.Series,
    smoothing_strength: int = 20,
) -> None:
    """Add an out-of-fold zipcode price prior."""

    global_log_price = np.log1p(train_prices).mean()
    target_frame = pd.DataFrame(
        {
            "zipcode": train_features["zipcode"],
            "log_price": np.log1p(train_prices),
        },
        index=train_features.index,
    )

    train_features["zipcode_log_price_prior"] = global_log_price
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for fit_positions, encode_positions in kfold.split(target_frame):
        fit_frame = target_frame.iloc[fit_positions]
        encode_index = target_frame.iloc[encode_positions].index

        fold_stats = fit_frame.groupby("zipcode")["log_price"].agg(["sum", "count"])
        fold_prior = (
            (fold_stats["sum"] + smoothing_strength * global_log_price)
            / (fold_stats["count"] + smoothing_strength)
        )

        train_features.loc[encode_index, "zipcode_log_price_prior"] = (
            train_features.loc[encode_index, "zipcode"].map(fold_prior).fillna(global_log_price)
        )

    full_stats = target_frame.groupby("zipcode")["log_price"].agg(["sum", "count"])
    full_prior = (
        (full_stats["sum"] + smoothing_strength * global_log_price)
        / (full_stats["count"] + smoothing_strength)
    )

    test_features["zipcode_log_price_prior"] = test_features["zipcode"].map(full_prior).fillna(global_log_price)

    for feature_frame in [train_features, test_features]:
        feature_frame["zipcode_train_count"] = feature_frame["zipcode"].map(full_stats["count"]).fillna(0)


def load_modeling_data(
    data_path: str = DATA_PATH,
    test_size: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, enrich, split, and encode the King County modeling data."""

    sales = parse_king_county_sales(data_path)
    target = sales["price"].astype(float)
    features = sales.drop(columns=["price"])

    # Why: a time-based split is closer to production pricing than a random
    # split: train on earlier sales, evaluate on future sales.
    split_index = int(len(sales) * (1 - test_size))
    train_features_raw = features.iloc[:split_index].copy()
    test_features_raw = features.iloc[split_index:].copy()
    train_prices = target.iloc[:split_index].copy()
    test_prices = target.iloc[split_index:].copy()

    train_features_raw, test_features_raw = add_king_county_features(
        train_features_raw,
        test_features_raw,
        train_prices,
    )

    train_features = pd.get_dummies(train_features_raw, dummy_na=True)
    test_features = pd.get_dummies(test_features_raw, dummy_na=True)
    test_features = test_features.reindex(columns=train_features.columns, fill_value=0)

    return train_features, test_features, train_prices, test_prices


def load_test_sale_context(
    data_path: str = DATA_PATH,
    test_size: float = 0.20,
) -> pd.DataFrame:
    """Return human-readable test-set property attributes for diagnostics."""

    sales = parse_king_county_sales(data_path)
    split_index = int(len(sales) * (1 - test_size))
    test_context = sales.iloc[split_index:].copy()

    test_context["actual_price"] = test_context["price"].astype(float)
    test_context["price_per_sqft"] = test_context["actual_price"] / test_context["sqft_living"].replace(0, np.nan)
    test_context["home_age"] = test_context["sale_date"].dt.year - test_context["yr_built"]
    test_context["was_renovated"] = (test_context["yr_renovated"] > 0).astype(int)
    test_context["lot_to_living_ratio"] = (
        test_context["sqft_lot"] / test_context["sqft_living"].replace(0, np.nan)
    )

    context_columns = [
        "sale_date",
        "actual_price",
        "price_per_sqft",
        "zipcode",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "lot_to_living_ratio",
        "waterfront",
        "view",
        "condition",
        "grade",
        "yr_built",
        "home_age",
        "was_renovated",
        "lat",
        "long",
    ]

    return test_context[context_columns]


def make_quantile_model(quantile: float, model_params: Dict[str, object] | None = None):
    """Create a LightGBM quantile regressor."""

    from lightgbm import LGBMRegressor

    model_params = model_params or {}

    return LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=model_params.get("n_estimators", 700),
        learning_rate=model_params.get("learning_rate", 0.035),
        num_leaves=model_params.get("num_leaves", 31),
        min_child_samples=model_params.get("min_child_samples", 80),
        subsample=model_params.get("subsample", 0.90),
        colsample_bytree=model_params.get("colsample_bytree", 0.90),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def tune_quantile_parameters(
    train_features: pd.DataFrame,
    train_prices: pd.Series,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Tune LightGBM settings using validation pinball loss and interval reliability."""

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
        {"n_estimators": 900, "learning_rate": 0.025, "num_leaves": 63, "min_child_samples": 120, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 700, "learning_rate": 0.035, "num_leaves": 63, "min_child_samples": 200, "subsample": 0.85, "colsample_bytree": 0.80},
    ]

    tuning_quantiles = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
    tuning_rows = []

    for candidate_index, candidate_params in enumerate(candidate_parameters, start=1):
        pinball_losses = []
        validation_prediction_frame = pd.DataFrame(index=validation_features.index)

        for quantile_name, quantile_level in tuning_quantiles.items():
            model = make_quantile_model(quantile_level, candidate_params)
            model.fit(model_features, model_prices)
            validation_predictions = model.predict(validation_features)
            validation_prediction_frame[quantile_name] = validation_predictions
            pinball_losses.append(mean_pinball_loss(validation_prices, validation_predictions, alpha=quantile_level))

        validation_prediction_frame[["q10", "q50", "q90"]] = np.sort(
            validation_prediction_frame[["q10", "q50", "q90"]].to_numpy(),
            axis=1,
        )

        q10_observed_frequency = (validation_prices <= validation_prediction_frame["q10"]).mean()
        q50_observed_frequency = (validation_prices <= validation_prediction_frame["q50"]).mean()
        q90_observed_frequency = (validation_prices <= validation_prediction_frame["q90"]).mean()
        interval_coverage = (
            (validation_prices >= validation_prediction_frame["q10"])
            & (validation_prices <= validation_prediction_frame["q90"])
        ).mean()

        quantile_calibration_error = np.mean(
            [
                abs(q10_observed_frequency - 0.10),
                abs(q50_observed_frequency - 0.50),
                abs(q90_observed_frequency - 0.90),
            ]
        )
        interval_coverage_error = abs(interval_coverage - 0.80)
        mean_pinball = float(np.mean(pinball_losses))
        selection_score = mean_pinball + 0.03 * quantile_calibration_error + 0.03 * interval_coverage_error

        tuning_rows.append(
            {
                "candidate": candidate_index,
                "selection_score": selection_score,
                "mean_pinball_loss": mean_pinball,
                "q10_observed_frequency": q10_observed_frequency,
                "q50_observed_frequency": q50_observed_frequency,
                "q90_observed_frequency": q90_observed_frequency,
                "q10_q90_interval_coverage": interval_coverage,
                **candidate_params,
            }
        )

    tuning_results = pd.DataFrame(tuning_rows).sort_values("selection_score")
    best_params = {key: tuning_results.iloc[0][key] for key in candidate_parameters[0].keys()}

    for integer_key in ["n_estimators", "num_leaves", "min_child_samples"]:
        best_params[integer_key] = int(best_params[integer_key])

    return best_params, tuning_results


def train_probabilistic_avm(
    train_features: pd.DataFrame,
    train_prices: pd.Series,
    test_features: pd.DataFrame,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, object], pd.DataFrame]:
    """Train quantile models from Q10 through Q90 and score the holdout set."""

    quantile_levels = {f"q{int(level * 100)}": level for level in np.arange(0.10, 1.00, 0.10)}
    trained_models: Dict[str, object] = {}
    prediction_frame = pd.DataFrame(index=test_features.index)
    best_model_params, tuning_results = tune_quantile_parameters(train_features, train_prices)

    log_train_prices = np.log1p(train_prices)

    for quantile_name, quantile_level in quantile_levels.items():
        model = make_quantile_model(quantile_level, best_model_params)
        model.fit(train_features, log_train_prices)

        trained_models[quantile_name] = model
        prediction_frame[quantile_name] = np.expm1(model.predict(test_features))

    quantile_columns = list(quantile_levels.keys())
    prediction_frame[quantile_columns] = np.sort(prediction_frame[quantile_columns].to_numpy(), axis=1)

    return trained_models, prediction_frame, best_model_params, tuning_results


def estimate_validation_predictions_for_calibration(
    train_features: pd.DataFrame,
    train_prices: pd.Series,
    model_params: Dict[str, object],
) -> Tuple[pd.Series, pd.DataFrame]:
    """Score the internal validation window so interval calibration stays leakage-safe."""

    log_train_prices = np.log1p(train_prices)
    model_features, validation_features, model_prices, validation_prices = train_test_split(
        train_features,
        log_train_prices,
        test_size=0.20,
        shuffle=False,
    )

    validation_prediction_frame = pd.DataFrame(index=validation_features.index)
    for quantile_name, quantile_level in {"q10": 0.10, "q50": 0.50, "q90": 0.90}.items():
        model = make_quantile_model(quantile_level, model_params)
        model.fit(model_features, model_prices)
        validation_prediction_frame[quantile_name] = np.expm1(model.predict(validation_features))

    validation_prediction_frame[["q10", "q50", "q90"]] = np.sort(
        validation_prediction_frame[["q10", "q50", "q90"]].to_numpy(),
        axis=1,
    )

    return np.expm1(validation_prices), validation_prediction_frame


def interval_coverage(actual_prices: pd.Series, prediction_frame: pd.DataFrame) -> float:
    """Measure how often actual prices land inside the Q10-Q90 interval."""

    return ((actual_prices >= prediction_frame["q10"]) & (actual_prices <= prediction_frame["q90"])).mean()


def widen_q10_q90_interval(prediction_frame: pd.DataFrame, calibration_factor: float) -> pd.DataFrame:
    """Widen Q10 and Q90 around Q50 while preserving the fair-value median."""

    calibrated_frame = prediction_frame.copy()
    calibrated_frame["q10"] = prediction_frame["q50"] - calibration_factor * (
        prediction_frame["q50"] - prediction_frame["q10"]
    )
    calibrated_frame["q90"] = prediction_frame["q50"] + calibration_factor * (
        prediction_frame["q90"] - prediction_frame["q50"]
    )
    calibrated_frame["q10"] = calibrated_frame["q10"].clip(lower=0)

    return calibrated_frame


def choose_width_calibration_factor(
    actual_prices: pd.Series,
    prediction_frame: pd.DataFrame,
    target_coverage: float = CALIBRATION_TARGET_COVERAGE,
    candidate_factors: np.ndarray = CALIBRATION_FACTOR_GRID,
) -> Tuple[float, pd.DataFrame]:
    """Choose the smallest interval-widening factor that reaches target coverage."""

    factor_rows = []
    for factor in candidate_factors:
        calibrated_frame = widen_q10_q90_interval(prediction_frame, factor)
        coverage = interval_coverage(actual_prices, calibrated_frame)
        factor_rows.append(
            {
                "calibration_factor": factor,
                "validation_coverage": coverage,
                "coverage_gap": coverage - target_coverage,
            }
        )

    factor_results = pd.DataFrame(factor_rows)
    target_hits = factor_results[factor_results["validation_coverage"] >= target_coverage]

    if target_hits.empty:
        best_factor = factor_results.sort_values("validation_coverage").iloc[-1]["calibration_factor"]
    else:
        # Why: once the interval is reliable enough, choose the narrowest version
        # so offers do not become more conservative than necessary.
        best_factor = target_hits.iloc[0]["calibration_factor"]

    return float(best_factor), factor_results


def calculate_buy_offer(
    median_price: pd.Series,
    lower_price: pd.Series,
    upper_price: pd.Series,
    spread_assumptions: SpreadAssumptions,
) -> pd.DataFrame:
    """Convert probabilistic valuation into a risk-adjusted acquisition offer."""

    offer_frame = pd.DataFrame(index=median_price.index)
    offer_frame["median_price"] = median_price
    offer_frame["uncertainty_width"] = upper_price - lower_price
    offer_frame["uncertainty_score"] = offer_frame["uncertainty_width"] / median_price
    offer_frame["base_spread"] = spread_assumptions.base_spread
    offer_frame["raw_uncertainty_penalty"] = spread_assumptions.penalty_strength * offer_frame["uncertainty_score"]
    offer_frame["uncertainty_penalty"] = offer_frame["raw_uncertainty_penalty"].clip(
        upper=spread_assumptions.max_uncertainty_penalty
    )
    offer_frame["hit_max_uncertainty_penalty"] = (
        offer_frame["raw_uncertainty_penalty"] >= spread_assumptions.max_uncertainty_penalty
    )
    offer_frame["total_spread"] = offer_frame["base_spread"] + offer_frame["uncertainty_penalty"]
    offer_frame["buy_offer"] = offer_frame["median_price"] * (1 - offer_frame["total_spread"])

    return offer_frame


def summarize_offer_policy(
    offer_frame: pd.DataFrame,
    spread_assumptions: SpreadAssumptions,
    calibration_factor: float,
) -> pd.Series:
    """Summarize the business knobs used by the offer engine."""

    return pd.Series(
        {
            "width_calibration_factor": calibration_factor,
            "transaction_cost_rate": spread_assumptions.transaction_cost_rate,
            "holding_cost_rate": spread_assumptions.holding_cost_rate,
            "resale_prep_cost_rate": spread_assumptions.resale_prep_cost_rate,
            "target_margin_rate": spread_assumptions.target_margin_rate,
            "base_spread": spread_assumptions.base_spread,
            "penalty_strength": spread_assumptions.penalty_strength,
            "max_uncertainty_penalty": spread_assumptions.max_uncertainty_penalty,
            "manual_review_candidate_count": int(offer_frame["hit_max_uncertainty_penalty"].sum()),
            "manual_review_candidate_share": offer_frame["hit_max_uncertainty_penalty"].mean(),
            "average_uncertainty_penalty": offer_frame["uncertainty_penalty"].mean(),
            "average_total_spread": offer_frame["total_spread"].mean(),
            "average_buy_offer": offer_frame["buy_offer"].mean(),
        }
    )


def summarize_pricing_results(
    actual_prices: pd.Series,
    prediction_frame: pd.DataFrame,
    offer_frame: pd.DataFrame,
) -> pd.Series:
    """Return headline metrics for model quality and offer behavior."""

    absolute_percentage_error = (actual_prices - prediction_frame["q50"]).abs() / actual_prices
    interval_coverage = (
        (actual_prices >= prediction_frame["q10"]) & (actual_prices <= prediction_frame["q90"])
    ).mean()

    return pd.Series(
        {
            "median_absolute_error": mean_absolute_error(actual_prices, prediction_frame["q50"]),
            "median_absolute_percentage_error": absolute_percentage_error.mean(),
            "root_mean_squared_error": root_mean_squared_error(actual_prices, prediction_frame["q50"]),
            "q10_q90_interval_coverage": interval_coverage,
            "expected_interval_coverage": 0.80,
            "average_interval_width_dollars": (prediction_frame["q90"] - prediction_frame["q10"]).mean(),
            "average_uncertainty_score": offer_frame["uncertainty_score"].mean(),
            "average_total_spread": offer_frame["total_spread"].mean(),
            "average_offer_to_value_ratio": (offer_frame["buy_offer"] / offer_frame["median_price"]).mean(),
        }
    )


def analyze_max_penalty_candidates(
    offer_frame: pd.DataFrame,
    test_context: pd.DataFrame,
    output_dir: Path = OFFER_OUTPUT_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Profile homes where the uncertainty penalty hits the policy cap."""

    output_dir.mkdir(exist_ok=True)

    candidate_frame = offer_frame.join(test_context, how="left")
    candidate_frame["offer_to_q50"] = candidate_frame["buy_offer"] / candidate_frame["median_price"]
    candidate_frame["offer_discount_to_actual"] = 1 - candidate_frame["buy_offer"] / candidate_frame["actual_price"]
    candidate_frame["cap_excess_penalty"] = (
        candidate_frame["raw_uncertainty_penalty"] - candidate_frame["uncertainty_penalty"]
    ).clip(lower=0)

    profile_rows = []
    for is_capped, group in candidate_frame.groupby("hit_max_uncertainty_penalty"):
        profile_rows.append(
            {
                "group": "hit_cap_manual_review" if is_capped else "below_cap_auto_price",
                "home_count": len(group),
                "share_of_test_set": len(group) / len(candidate_frame),
                "median_actual_price": group["actual_price"].median(),
                "median_predicted_q50": group["median_price"].median(),
                "median_buy_offer": group["buy_offer"].median(),
                "average_offer_to_q50": group["offer_to_q50"].mean(),
                "average_uncertainty_score": group["uncertainty_score"].mean(),
                "median_sqft_living": group["sqft_living"].median(),
                "median_sqft_lot": group["sqft_lot"].median(),
                "average_grade": group["grade"].mean(),
                "average_home_age": group["home_age"].mean(),
                "waterfront_rate": group["waterfront"].mean(),
                "view_rate": (group["view"] > 0).mean(),
                "renovated_rate": group["was_renovated"].mean(),
            }
        )

    profile_summary = pd.DataFrame(profile_rows)

    capped_candidates = candidate_frame[candidate_frame["hit_max_uncertainty_penalty"]].copy()
    zipcode_summary = (
        capped_candidates.groupby("zipcode")
        .agg(
            capped_home_count=("actual_price", "size"),
            median_actual_price=("actual_price", "median"),
            median_sqft_living=("sqft_living", "median"),
            average_uncertainty_score=("uncertainty_score", "mean"),
        )
        .sort_values(["capped_home_count", "average_uncertainty_score"], ascending=False)
        .head(15)
        .reset_index()
    )

    candidate_frame.to_csv(output_dir / "manual_review_candidate_details.csv", index=True, index_label="row_id")
    profile_summary.to_csv(output_dir / "manual_review_candidate_summary.csv", index=False)
    zipcode_summary.to_csv(output_dir / "manual_review_candidate_zipcodes.csv", index=False)

    return candidate_frame, profile_summary, zipcode_summary


def evaluate_uncertainty_signal(
    actual_prices: pd.Series,
    prediction_frame: pd.DataFrame,
    offer_frame: pd.DataFrame,
    bucket_count: int = 3,
) -> pd.DataFrame:
    """Check whether wider prediction bands correspond to larger errors."""

    evaluation_frame = pd.DataFrame(index=actual_prices.index)
    evaluation_frame["actual_price"] = actual_prices
    evaluation_frame["predicted_median"] = prediction_frame["q50"]
    evaluation_frame["uncertainty_score"] = offer_frame["uncertainty_score"]
    evaluation_frame["absolute_error"] = (
        evaluation_frame["actual_price"] - evaluation_frame["predicted_median"]
    ).abs()
    evaluation_frame["absolute_percentage_error"] = (
        evaluation_frame["absolute_error"] / evaluation_frame["actual_price"]
    )
    evaluation_frame["covered_by_q10_q90"] = (
        (evaluation_frame["actual_price"] >= prediction_frame["q10"])
        & (evaluation_frame["actual_price"] <= prediction_frame["q90"])
    )

    evaluation_frame["uncertainty_bucket"] = pd.qcut(
        evaluation_frame["uncertainty_score"],
        q=bucket_count,
        labels=["low", "medium", "high"][:bucket_count],
        duplicates="drop",
    )

    return (
        evaluation_frame.groupby("uncertainty_bucket", observed=True)
        .agg(
            home_count=("actual_price", "size"),
            average_uncertainty_score=("uncertainty_score", "mean"),
            median_absolute_error=("absolute_error", "median"),
            mean_absolute_percentage_error=("absolute_percentage_error", "mean"),
            q10_q90_interval_coverage=("covered_by_q10_q90", "mean"),
        )
        .reset_index()
    )


def plot_interval_calibration(
    summary: pd.Series,
    uncertainty_bucket_summary: pd.DataFrame,
    output_path: Path = MODEL_OUTPUT_DIR / "calibration_check.png",
) -> None:
    """Plot expected interval coverage against observed coverage."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_coverage = summary["expected_interval_coverage"]
    overall_coverage = summary["q10_q90_interval_coverage"]
    bucket_labels = uncertainty_bucket_summary["uncertainty_bucket"].astype(str)
    bucket_coverages = uncertainty_bucket_summary["q10_q90_interval_coverage"]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    axes[0].bar(["Expected", "Observed"], [expected_coverage, overall_coverage], color=["#4C78A8", "#F58518"])
    axes[0].set_title("Overall Q10-Q90 Coverage")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Share of Actual Prices Covered")
    axes[0].axhline(expected_coverage, color="#4C78A8", linestyle="--", linewidth=1)

    for index, value in enumerate([expected_coverage, overall_coverage]):
        axes[0].text(index, value + 0.03, f"{value:.1%}", ha="center", fontsize=10)

    axes[1].bar(bucket_labels, bucket_coverages, color="#54A24B")
    axes[1].axhline(expected_coverage, color="#4C78A8", linestyle="--", linewidth=1, label="Expected 80%")
    axes[1].set_title("Coverage by Uncertainty Bucket")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Share of Actual Prices Covered")
    axes[1].legend(frameon=False)

    for index, value in enumerate(bucket_coverages):
        axes[1].text(index, value + 0.03, f"{value:.1%}", ha="center", fontsize=10)

    figure.suptitle("Calibration Check: Are Prediction Intervals Wide Enough?", fontsize=13)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_quantile_calibration(
    actual_prices: pd.Series,
    prediction_frame: pd.DataFrame,
    output_path: Path = MODEL_OUTPUT_DIR / "quantile_calibration_curve.png",
) -> pd.DataFrame:
    """Plot predicted quantile levels against observed frequencies."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_rows = []
    for quantile_level in np.arange(0.10, 1.00, 0.10):
        quantile_column = f"q{int(quantile_level * 100)}"
        observed_frequency = (actual_prices <= prediction_frame[quantile_column]).mean()
        calibration_rows.append(
            {
                "predicted_quantile": quantile_level,
                "observed_frequency": observed_frequency,
                "calibration_error": observed_frequency - quantile_level,
            }
        )

    calibration_frame = pd.DataFrame(calibration_rows)

    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="#4C78A8", linestyle="--", label="Perfect calibration")
    axis.plot(
        calibration_frame["predicted_quantile"],
        calibration_frame["observed_frequency"],
        marker="o",
        color="#F58518",
        label="Observed frequency",
    )
    axis.set_title("Quantile Calibration Curve")
    axis.set_xlabel("Predicted Quantile")
    axis.set_ylabel("Actual Observed Frequency")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)

    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return calibration_frame


def plot_error_by_uncertainty_bucket(
    uncertainty_bucket_summary: pd.DataFrame,
    output_path: Path = MODEL_OUTPUT_DIR / "error_by_uncertainty_bucket.png",
) -> None:
    """Plot whether homes with wider intervals have larger realized errors."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    bucket_labels = uncertainty_bucket_summary["uncertainty_bucket"].astype(str)
    error_rates = uncertainty_bucket_summary["mean_absolute_percentage_error"]

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.bar(bucket_labels, error_rates, color=["#72B7B2", "#F2CF5B", "#E45756"])
    axis.set_title("Prediction Error Increases With Model Uncertainty")
    axis.set_xlabel("Uncertainty Bucket")
    axis.set_ylabel("Mean Absolute Percentage Error")
    axis.set_ylim(0, max(error_rates.max() * 1.25, 0.20))

    for index, value in enumerate(error_rates):
        axis.text(index, value + 0.005, f"{value:.1%}", ha="center", fontsize=10)

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_offer_spread_by_uncertainty_bucket(
    uncertainty_bucket_summary: pd.DataFrame,
    offer_frame: pd.DataFrame,
    output_path: Path = OFFER_OUTPUT_DIR / "offer_spread_by_uncertainty_bucket.png",
) -> None:
    """Show how the calibrated uncertainty score turns into wider offer spreads."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotting_frame = offer_frame.copy()
    plotting_frame["uncertainty_bucket"] = pd.qcut(
        plotting_frame["uncertainty_score"],
        q=len(uncertainty_bucket_summary),
        labels=uncertainty_bucket_summary["uncertainty_bucket"].astype(str).tolist(),
        duplicates="drop",
    )
    plotting_frame["offer_to_value"] = plotting_frame["buy_offer"] / plotting_frame["median_price"]
    spread_by_bucket = (
        plotting_frame.groupby("uncertainty_bucket", observed=True)
        .agg(
            average_base_spread=("base_spread", "mean"),
            average_uncertainty_penalty=("uncertainty_penalty", "mean"),
            average_total_spread=("total_spread", "mean"),
            average_offer_to_value=("offer_to_value", "mean"),
        )
        .reset_index()
    )

    bucket_labels = spread_by_bucket["uncertainty_bucket"].astype(str)
    base_spread = spread_by_bucket["average_base_spread"]
    uncertainty_penalty = spread_by_bucket["average_uncertainty_penalty"]

    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.bar(bucket_labels, base_spread, color="#4C78A8", label="Base spread")
    axis.bar(bucket_labels, uncertainty_penalty, bottom=base_spread, color="#F58518", label="Uncertainty penalty")
    axis.set_title("Offer Spread Widens For Higher-Uncertainty Homes")
    axis.set_xlabel("Uncertainty Bucket")
    axis.set_ylabel("Average Offer Spread")
    axis.set_ylim(0, max(spread_by_bucket["average_total_spread"].max() * 1.25, 0.16))
    axis.legend(frameon=False)

    for index, value in enumerate(spread_by_bucket["average_total_spread"]):
        axis.text(index, value + 0.004, f"{value:.1%}", ha="center", fontsize=10)

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_manual_review_candidate_profile(
    profile_summary: pd.DataFrame,
    output_path: Path = OFFER_OUTPUT_DIR / "manual_review_candidate_profile.png",
) -> None:
    """Compare capped manual-review candidates against normal auto-priced homes."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_frame = profile_summary.set_index("group").loc[
        ["below_cap_auto_price", "hit_cap_manual_review"]
    ]
    labels = ["Below cap", "Hit cap"]
    metrics = [
        ("median_actual_price", "Median Actual Price", "Dollars"),
        ("median_sqft_living", "Median Living Area", "Sqft"),
        ("median_sqft_lot", "Median Lot Size", "Sqft"),
        ("average_grade", "Average Grade", "Grade"),
        ("waterfront_rate", "Waterfront Share", "Share"),
        ("view_rate", "Any View Share", "Share"),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes.ravel(), metrics):
        values = plot_frame[metric]
        axis.bar(labels, values, color=["#4C78A8", "#E45756"])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)

        for index, value in enumerate(values):
            if "rate" in metric:
                text = f"{value:.1%}"
            elif metric.startswith("median_actual_price"):
                text = f"${value / 1000:.0f}K"
            elif metric.startswith("median_sqft"):
                text = f"{value:,.0f}"
            else:
                text = f"{value:.1f}"
            axis.text(index, value * 1.02 if value else 0.02, text, ha="center", fontsize=9)

    figure.suptitle("Homes Hitting Max Uncertainty Penalty Look Different", fontsize=13)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_manual_review_candidate_report(
    profile_summary: pd.DataFrame,
    zipcode_summary: pd.DataFrame,
    output_path: Path = OFFER_OUTPUT_DIR / "manual_review_candidate_report.md",
) -> None:
    """Write a business-readable summary of max-penalty homes."""

    capped_profile = profile_summary.set_index("group").loc["hit_cap_manual_review"]
    auto_profile = profile_summary.set_index("group").loc["below_cap_auto_price"]

    report = f"""# Manual Review Candidate Analysis

The offer engine caps the uncertainty penalty at 10%. Homes that hit this cap
are useful to inspect because they may be better suited for manual review than
fully automated pricing.

## Headline

| Metric | Below Cap | Hit Cap |
|---|---:|---:|
| Home count | {auto_profile['home_count']:.0f} | {capped_profile['home_count']:.0f} |
| Share of test set | {auto_profile['share_of_test_set']:.1%} | {capped_profile['share_of_test_set']:.1%} |
| Avg uncertainty score | {auto_profile['average_uncertainty_score']:.1%} | {capped_profile['average_uncertainty_score']:.1%} |
| Median actual price | ${auto_profile['median_actual_price']:,.0f} | ${capped_profile['median_actual_price']:,.0f} |
| Median living area | {auto_profile['median_sqft_living']:,.0f} sqft | {capped_profile['median_sqft_living']:,.0f} sqft |
| Median lot size | {auto_profile['median_sqft_lot']:,.0f} sqft | {capped_profile['median_sqft_lot']:,.0f} sqft |
| Average grade | {auto_profile['average_grade']:.2f} | {capped_profile['average_grade']:.2f} |
| Waterfront share | {auto_profile['waterfront_rate']:.1%} | {capped_profile['waterfront_rate']:.1%} |
| Any-view share | {auto_profile['view_rate']:.1%} | {capped_profile['view_rate']:.1%} |

![Manual review candidate profile](manual_review_candidate_profile.png)

## Interpretation

The max-penalty homes are the cases where the model's calibrated uncertainty
would imply an even larger penalty, but the business policy caps the discount.
That makes them natural candidates for a manual review queue, a more detailed
inspection workflow, or a no-instant-offer rule.

## Top Zipcodes Among Capped Homes

{zipcode_summary.to_markdown(index=False)}
"""

    output_path.write_text(report)


def main() -> None:
    """Run the King County probabilistic AVM plus buy-offer engine."""

    # Step 1: Load King County sales, join monthly FRED macro features, create
    # real-estate features, and split earlier sales for training vs later sales
    # for testing. This mimics pricing future homes from historical data.
    train_features, test_features, train_prices, test_prices = load_modeling_data()

    # Step 2: Tune LightGBM quantile settings, then train Q10 through Q90
    # models. Q50 is the fair-value estimate; Q10-Q90 defines the uncertainty
    # band around that fair value.
    _, prediction_frame, best_model_params, tuning_results = train_probabilistic_avm(
        train_features, train_prices, test_features
    )

    # Step 3: Learn a width calibration factor on the internal validation
    # window. Raw quantile models rank risk well but are usually overconfident,
    # so this widens Q10-Q90 before the interval width becomes offer spread.
    validation_prices, validation_prediction_frame = estimate_validation_predictions_for_calibration(
        train_features,
        train_prices,
        best_model_params,
    )
    calibration_factor, calibration_factor_results = choose_width_calibration_factor(
        validation_prices,
        validation_prediction_frame,
    )
    calibrated_prediction_frame = widen_q10_q90_interval(prediction_frame, calibration_factor)

    # Step 4: Convert the calibrated probabilistic valuation into an acquisition
    # offer. Q50 remains fair value; calibrated Q10-Q90 defines the uncertainty
    # penalty that gets added on top of the base business spread.
    spread_assumptions = SpreadAssumptions()
    offer_frame = calculate_buy_offer(
        median_price=calibrated_prediction_frame["q50"],
        lower_price=calibrated_prediction_frame["q10"],
        upper_price=calibrated_prediction_frame["q90"],
        spread_assumptions=spread_assumptions,
    )
    offer_policy_summary = summarize_offer_policy(offer_frame, spread_assumptions, calibration_factor)
    test_context = load_test_sale_context()
    manual_review_details, manual_review_summary, manual_review_zipcodes = analyze_max_penalty_candidates(
        offer_frame,
        test_context,
    )

    # Step 5: Evaluate the AVM as both a price model and a risk signal. We care
    # about median accuracy, calibrated interval coverage, and whether wider
    # intervals actually correspond to larger realized pricing errors.
    summary = summarize_pricing_results(test_prices, calibrated_prediction_frame, offer_frame)
    raw_summary = summarize_pricing_results(
        test_prices,
        prediction_frame,
        calculate_buy_offer(
            prediction_frame["q50"],
            prediction_frame["q10"],
            prediction_frame["q90"],
            spread_assumptions,
        ),
    )
    uncertainty_bucket_summary = evaluate_uncertainty_signal(
        test_prices,
        calibrated_prediction_frame,
        offer_frame,
    )

    print("\nHeadline model and calibrated offer metrics:")
    print(summary.round(4))

    print("\nOffer policy assumptions:")
    print(offer_policy_summary.round(4))

    print("\nHow many homes hit the max uncertainty penalty?")
    print(manual_review_summary.round(4).to_string(index=False))

    print("\nRaw vs calibrated Q10-Q90 coverage:")
    print(
        pd.Series(
            {
                "raw_q10_q90_coverage": raw_summary["q10_q90_interval_coverage"],
                "calibrated_q10_q90_coverage": summary["q10_q90_interval_coverage"],
                "target_coverage": CALIBRATION_TARGET_COVERAGE,
            }
        ).round(4)
    )

    print("\nDoes model uncertainty predict actual pricing error?")
    print(uncertainty_bucket_summary.round(4).to_string(index=False))

    # Step 6: Save decision-focused diagnostic plots. The first plot checks
    # whether the Q10-Q90 interval covers the expected 80% of outcomes. The
    # second checks calibration at every quantile from Q10 through Q90.
    plot_interval_calibration(summary, uncertainty_bucket_summary)
    print(f"\nSaved calibration plot to {MODEL_OUTPUT_DIR / 'calibration_check.png'}")

    quantile_calibration = plot_quantile_calibration(test_prices, prediction_frame)
    print(f"Saved quantile calibration curve to {MODEL_OUTPUT_DIR / 'quantile_calibration_curve.png'}")

    # Step 6b: Save the core risk-ranking plot. This shows whether wider
    # prediction intervals identify homes where the median AVM is actually more
    # likely to be wrong.
    plot_error_by_uncertainty_bucket(uncertainty_bucket_summary)
    print(f"Saved error-by-uncertainty plot to {MODEL_OUTPUT_DIR / 'error_by_uncertainty_bucket.png'}")

    plot_offer_spread_by_uncertainty_bucket(uncertainty_bucket_summary, offer_frame)
    print(f"Saved offer-spread plot to {OFFER_OUTPUT_DIR / 'offer_spread_by_uncertainty_bucket.png'}")

    plot_manual_review_candidate_profile(manual_review_summary)
    write_manual_review_candidate_report(manual_review_summary, manual_review_zipcodes)
    print(f"Saved manual-review candidate report to {OFFER_OUTPUT_DIR / 'manual_review_candidate_report.md'}")

    print("\nBest LightGBM parameters from validation tuning:")
    print(pd.Series(best_model_params).to_string())

    # Step 7: Persist the core outputs so the notebook/deck can reuse the exact
    # same metrics and plots without rerunning the full training pipeline.
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OFFER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tuning_results.to_csv(MODEL_OUTPUT_DIR / "lightgbm_tuning_results.csv", index=False)
    summary.to_csv(MODEL_OUTPUT_DIR / "model_summary_metrics.csv", header=["value"])
    raw_summary.to_csv(MODEL_OUTPUT_DIR / "raw_model_summary_metrics.csv", header=["value"])
    calibration_factor_results.to_csv(MODEL_OUTPUT_DIR / "width_calibration_factor_search.csv", index=False)
    uncertainty_bucket_summary.to_csv(MODEL_OUTPUT_DIR / "uncertainty_bucket_summary.csv", index=False)
    quantile_calibration.to_csv(MODEL_OUTPUT_DIR / "quantile_calibration_table.csv", index=False)

    offer_policy_summary.to_csv(OFFER_OUTPUT_DIR / "offer_policy_summary.csv", header=["value"])
    offer_frame.to_csv(OFFER_OUTPUT_DIR / "buy_offer_recommendations.csv", index=True, index_label="row_id")

    print(f"Saved tuning results to {MODEL_OUTPUT_DIR / 'lightgbm_tuning_results.csv'}")

    print("\nQuantile calibration table:")
    print(quantile_calibration.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
