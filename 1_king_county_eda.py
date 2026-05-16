"""Exploratory data analysis for the King County pricing project.

This script is intentionally separate from the modeling pipeline. It studies
the housing and macro environment before we build the risk-aware pricing model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = "kc_house_data.csv"
MACRO_PATH = Path("outputs/3_data/fred_macro_features.csv")
EDA_OUTPUT_DIR = Path("outputs/1_eda")
TRAIN_FRACTION = 0.64
VALIDATION_FRACTION = 0.16
TEST_FRACTION = 0.20


def load_sales_with_macro() -> pd.DataFrame:
    """Load King County sales and attach cached monthly macro features."""

    sales = pd.read_csv(DATA_PATH)
    sales["sale_date"] = pd.to_datetime(sales["date"], format="%Y%m%dT%H%M%S")
    sales["sale_month"] = sales["sale_date"].values.astype("datetime64[M]")

    macro_features = pd.read_csv(MACRO_PATH, parse_dates=["sale_month"])
    sales = sales.merge(macro_features, on="sale_month", how="left")

    return sales.sort_values("sale_date").reset_index(drop=True)


def assign_model_splits(sales: pd.DataFrame) -> pd.DataFrame:
    """Assign the same train/validation/test periods used by the model."""

    sales = sales.sort_values("sale_date").reset_index(drop=True).copy()
    train_end = int(len(sales) * TRAIN_FRACTION)
    validation_end = int(len(sales) * (TRAIN_FRACTION + VALIDATION_FRACTION))

    sales["model_split"] = "test"
    sales.loc[: train_end - 1, "model_split"] = "train"
    sales.loc[train_end : validation_end - 1, "model_split"] = "validation"

    return sales


def get_split_boundaries(sales: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Return boundary dates where validation and test periods begin."""

    return {
        "validation_start": sales.loc[sales["model_split"] == "validation", "sale_date"].min(),
        "test_start": sales.loc[sales["model_split"] == "test", "sale_date"].min(),
    }


def add_split_markers(axis: plt.Axes, split_boundaries: dict[str, pd.Timestamp]) -> None:
    """Draw vertical split markers on time-series plots."""

    marker_specs = {
        "validation_start": ("Validation starts", "#B279A2"),
        "test_start": ("Test starts", "#E45756"),
    }

    for boundary_name, (label, color) in marker_specs.items():
        boundary_date = split_boundaries[boundary_name]
        axis.axvline(boundary_date, color=color, linestyle="--", linewidth=1.5, alpha=0.90, label=label)


def add_single_split_legend(axis: plt.Axes) -> None:
    """Keep train/validation/test marker legends from duplicating too much."""

    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), frameon=False)


def summarize_monthly_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Create monthly sales, price, and inventory-style summaries."""

    monthly = (
        sales.groupby("sale_month")
        .agg(
            sale_count=("price", "size"),
            median_sale_price=("price", "median"),
            average_sale_price=("price", "mean"),
            p25_sale_price=("price", lambda values: values.quantile(0.25)),
            p75_sale_price=("price", lambda values: values.quantile(0.75)),
            median_price_per_sqft=("price_per_sqft", "median"),
            median_sqft_living=("sqft_living", "median"),
        )
        .reset_index()
    )

    macro_columns = [column for column in sales.columns if column not in monthly.columns]
    macro_columns = [
        column
        for column in macro_columns
        if column
        in {
            "mortgage_30y_rate",
            "case_shiller_seattle",
            "seattle_unemployment_rate",
            "treasury_10y_rate",
            "consumer_sentiment",
            "months_supply_homes",
            "housing_starts",
            "consumer_price_index",
        }
    ]
    monthly_macro = sales.groupby("sale_month")[macro_columns].mean().reset_index()

    return monthly.merge(monthly_macro, on="sale_month", how="left")


def plot_sales_volume_and_prices(monthly: pd.DataFrame, split_boundaries: dict[str, pd.Timestamp]) -> None:
    """Show how transaction volume and sale prices changed over time."""

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)

    axes[0].bar(monthly["sale_month"], monthly["sale_count"], width=22, color="#4C78A8")
    axes[0].set_title("King County Monthly Sales Volume")
    axes[0].set_ylabel("Homes Sold")
    add_split_markers(axes[0], split_boundaries)
    add_single_split_legend(axes[0])

    axes[1].plot(monthly["sale_month"], monthly["median_sale_price"], marker="o", label="Median sale price")
    axes[1].fill_between(
        monthly["sale_month"],
        monthly["p25_sale_price"],
        monthly["p75_sale_price"],
        color="#F58518",
        alpha=0.20,
        label="25th-75th percentile",
    )
    axes[1].set_title("Monthly Sale Price Distribution")
    axes[1].set_ylabel("Sale Price ($)")
    add_split_markers(axes[1], split_boundaries)
    add_single_split_legend(axes[1])

    fig.savefig(EDA_OUTPUT_DIR / "monthly_sales_volume_and_prices.png", dpi=160)
    plt.close(fig)


def plot_price_per_sqft(monthly: pd.DataFrame, split_boundaries: dict[str, pd.Timestamp]) -> None:
    """Track normalized price movement using median price per square foot."""

    fig, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    axis.plot(monthly["sale_month"], monthly["median_price_per_sqft"], marker="o", color="#54A24B")
    axis.set_title("Median Price Per Square Foot Over Time")
    axis.set_ylabel("Median $ / Sq Ft")
    axis.set_xlabel("Sale Month")
    add_split_markers(axis, split_boundaries)
    add_single_split_legend(axis)

    fig.savefig(EDA_OUTPUT_DIR / "monthly_price_per_sqft.png", dpi=160)
    plt.close(fig)


def plot_macro_environment(monthly: pd.DataFrame, split_boundaries: dict[str, pd.Timestamp]) -> None:
    """Show the macro backdrop during the King County sale window."""

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)

    axes[0].plot(monthly["sale_month"], monthly["mortgage_30y_rate"], marker="o", color="#4C78A8")
    axes[0].set_title("30-Year Mortgage Rate")
    axes[0].set_ylabel("Rate (%)")
    add_split_markers(axes[0], split_boundaries)
    add_single_split_legend(axes[0])

    axes[1].plot(monthly["sale_month"], monthly["case_shiller_seattle"], marker="o", color="#F58518")
    axes[1].set_title("Seattle Case-Shiller Home Price Index")
    axes[1].set_ylabel("Index")
    add_split_markers(axes[1], split_boundaries)

    axes[2].plot(monthly["sale_month"], monthly["seattle_unemployment_rate"], marker="o", color="#E45756")
    axes[2].set_title("Seattle-Area Unemployment Rate")
    axes[2].set_ylabel("Rate (%)")
    add_split_markers(axes[2], split_boundaries)

    axes[3].plot(monthly["sale_month"], monthly["consumer_sentiment"], marker="o", color="#72B7B2")
    axes[3].set_title("Consumer Sentiment")
    axes[3].set_ylabel("Index")
    axes[3].set_xlabel("Sale Month")
    add_split_markers(axes[3], split_boundaries)

    fig.savefig(EDA_OUTPUT_DIR / "macro_environment_over_time.png", dpi=160)
    plt.close(fig)


def plot_rates_and_supply(monthly: pd.DataFrame, split_boundaries: dict[str, pd.Timestamp]) -> None:
    """Compare financing rates and national housing supply signals."""

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)

    axes[0].plot(monthly["sale_month"], monthly["mortgage_30y_rate"], marker="o", label="30Y mortgage")
    axes[0].plot(monthly["sale_month"], monthly["treasury_10y_rate"], marker="o", label="10Y treasury")
    axes[0].set_title("Rates Environment")
    axes[0].set_ylabel("Rate (%)")
    add_split_markers(axes[0], split_boundaries)
    add_single_split_legend(axes[0])

    axes[1].plot(monthly["sale_month"], monthly["months_supply_homes"], marker="o", color="#B279A2")
    axes[1].set_title("Months Supply of Homes")
    axes[1].set_ylabel("Months")
    add_split_markers(axes[1], split_boundaries)

    axes[2].plot(monthly["sale_month"], monthly["housing_starts"], marker="o", color="#FF9DA6")
    axes[2].set_title("Housing Starts")
    axes[2].set_ylabel("Thousands")
    axes[2].set_xlabel("Sale Month")
    add_split_markers(axes[2], split_boundaries)

    fig.savefig(EDA_OUTPUT_DIR / "rates_supply_and_starts.png", dpi=160)
    plt.close(fig)


def plot_price_distribution(sales: pd.DataFrame) -> None:
    """Show the spread and skew in King County sale prices."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axes[0].hist(sales["price"], bins=60, color="#4C78A8")
    axes[0].set_title("Sale Price Distribution")
    axes[0].set_xlabel("Sale Price ($)")
    axes[0].set_ylabel("Home Count")

    axes[1].hist(sales["price_per_sqft"], bins=60, color="#F58518")
    axes[1].set_title("Price Per Square Foot Distribution")
    axes[1].set_xlabel("$ / Sq Ft")

    fig.savefig(EDA_OUTPUT_DIR / "price_distribution.png", dpi=160)
    plt.close(fig)


def plot_split_price_distribution(sales: pd.DataFrame) -> None:
    """Compare price and price-per-square-foot distributions by model split."""

    split_order = ["train", "validation", "test"]
    colors = ["#4C78A8", "#B279A2", "#E45756"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    axes[0].boxplot(
        [sales.loc[sales["model_split"] == split, "price"] for split in split_order],
        tick_labels=split_order,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
    )
    for patch, color in zip(axes[0].artists, colors):
        patch.set_facecolor(color)
    axes[0].set_title("Sale Price Distribution by Model Split")
    axes[0].set_ylabel("Sale Price ($)")

    axes[1].boxplot(
        [sales.loc[sales["model_split"] == split, "price_per_sqft"] for split in split_order],
        tick_labels=split_order,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
    )
    for patch, color in zip(axes[1].artists, colors):
        patch.set_facecolor(color)
    axes[1].set_title("Price Per Sq Ft Distribution by Model Split")
    axes[1].set_ylabel("$ / Sq Ft")

    fig.savefig(EDA_OUTPUT_DIR / "split_price_distribution.png", dpi=160)
    plt.close(fig)


def plot_split_property_profile(sales: pd.DataFrame) -> None:
    """Compare property mix across train, validation, and test periods."""

    split_profile = (
        sales.groupby("model_split")
        .agg(
            median_price=("price", "median"),
            median_sqft_living=("sqft_living", "median"),
            median_grade=("grade", "median"),
            waterfront_share=("waterfront", "mean"),
            renovated_share=("yr_renovated", lambda values: (values > 0).mean()),
            median_price_per_sqft=("price_per_sqft", "median"),
        )
        .loc[["train", "validation", "test"]]
    )

    plot_columns = [
        "median_price",
        "median_sqft_living",
        "median_grade",
        "waterfront_share",
        "renovated_share",
        "median_price_per_sqft",
    ]
    titles = [
        "Median Price",
        "Median Living Sq Ft",
        "Median Grade",
        "Waterfront Share",
        "Renovated Share",
        "Median $ / Sq Ft",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, column, title in zip(axes.ravel(), plot_columns, titles):
        axis.bar(split_profile.index, split_profile[column], color=["#4C78A8", "#B279A2", "#E45756"])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)

    fig.savefig(EDA_OUTPUT_DIR / "split_property_profile.png", dpi=160)
    plt.close(fig)

    split_profile.to_csv(EDA_OUTPUT_DIR / "split_property_profile.csv")


def plot_top_zipcode_mix(sales: pd.DataFrame) -> None:
    """Show whether later splits contain a different zipcode mix."""

    top_zipcodes = sales["zipcode"].value_counts().head(10).index
    zipcode_counts = (
        sales[sales["zipcode"].isin(top_zipcodes)]
        .groupby(["model_split", "zipcode"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["train", "validation", "test"])
    )

    zipcode_mix = zipcode_counts.div(zipcode_counts.sum(axis=1), axis=0)
    pivot = zipcode_mix.T.reindex(index=top_zipcodes)

    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x_positions = np.arange(len(pivot.index))
    width = 0.25

    for offset, split, color in zip([-width, 0, width], ["train", "validation", "test"], ["#4C78A8", "#B279A2", "#E45756"]):
        axis.bar(x_positions + offset, pivot[split], width=width, label=split, color=color)

    axis.set_title("Top Zipcode Mix by Model Split")
    axis.set_xlabel("Zipcode")
    axis.set_ylabel("Share Within Top 10 Zipcodes")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(pivot.index.astype(str), rotation=30)
    axis.legend(frameon=False)

    fig.savefig(EDA_OUTPUT_DIR / "top_zipcode_mix_by_split.png", dpi=160)
    plt.close(fig)

    pivot.to_csv(EDA_OUTPUT_DIR / "top_zipcode_mix_by_split.csv")


def save_summary_tables(sales: pd.DataFrame, monthly: pd.DataFrame) -> None:
    """Save compact tables that can be referenced in the presentation."""

    summary = pd.Series(
        {
            "sale_start_date": sales["sale_date"].min().strftime("%Y-%m-%d"),
            "sale_end_date": sales["sale_date"].max().strftime("%Y-%m-%d"),
            "home_count": len(sales),
            "median_sale_price": sales["price"].median(),
            "average_sale_price": sales["price"].mean(),
            "median_price_per_sqft": sales["price_per_sqft"].median(),
            "median_sqft_living": sales["sqft_living"].median(),
            "waterfront_share": sales["waterfront"].mean(),
        }
    )

    summary.to_csv(EDA_OUTPUT_DIR / "dataset_summary.csv", header=["value"])
    monthly.to_csv(EDA_OUTPUT_DIR / "monthly_sales_macro_summary.csv", index=False)

    split_summary = (
        sales.groupby("model_split")
        .agg(
            start_date=("sale_date", "min"),
            end_date=("sale_date", "max"),
            home_count=("price", "size"),
            median_sale_price=("price", "median"),
            average_sale_price=("price", "mean"),
            median_price_per_sqft=("price_per_sqft", "median"),
            median_sqft_living=("sqft_living", "median"),
            waterfront_share=("waterfront", "mean"),
            renovated_share=("yr_renovated", lambda values: (values > 0).mean()),
        )
        .loc[["train", "validation", "test"]]
    )
    split_summary.to_csv(EDA_OUTPUT_DIR / "split_summary.csv")


def write_eda_report(sales: pd.DataFrame, monthly: pd.DataFrame, split_boundaries: dict[str, pd.Timestamp]) -> None:
    """Create a lightweight Markdown report that stitches EDA outputs together."""

    summary = pd.read_csv(EDA_OUTPUT_DIR / "dataset_summary.csv", index_col=0)["value"]
    split_summary = pd.read_csv(EDA_OUTPUT_DIR / "split_summary.csv", index_col=0)

    start_date = summary["sale_start_date"]
    end_date = summary["sale_end_date"]
    median_price = float(summary["median_sale_price"])
    median_ppsf = float(summary["median_price_per_sqft"])

    price_per_sqft_start = monthly["median_price_per_sqft"].iloc[0]
    price_per_sqft_end = monthly["median_price_per_sqft"].iloc[-1]
    ppsf_change = (price_per_sqft_end / price_per_sqft_start) - 1

    case_shiller_start = monthly["case_shiller_seattle"].iloc[0]
    case_shiller_end = monthly["case_shiller_seattle"].iloc[-1]
    case_shiller_change = (case_shiller_end / case_shiller_start) - 1

    mortgage_start = monthly["mortgage_30y_rate"].iloc[0]
    mortgage_end = monthly["mortgage_30y_rate"].iloc[-1]

    report = f"""# King County Housing And Macro EDA

This pre-study uses the full King County dataset and the attached FRED macro
features before the probabilistic AVM is trained.

## Dataset Window

- Sale window: `{start_date}` to `{end_date}`
- Homes: `{int(float(summary['home_count'])):,}`
- Median sale price: `${median_price:,.0f}`
- Median price per square foot: `${median_ppsf:,.0f}`

The model split is time based:

| Split | Start | End | Homes | Median Price | Median $/Sq Ft |
|---|---:|---:|---:|---:|---:|
| Train | {split_summary.loc['train', 'start_date'][:10]} | {split_summary.loc['train', 'end_date'][:10]} | {int(split_summary.loc['train', 'home_count']):,} | ${split_summary.loc['train', 'median_sale_price']:,.0f} | ${split_summary.loc['train', 'median_price_per_sqft']:,.0f} |
| Validation | {split_summary.loc['validation', 'start_date'][:10]} | {split_summary.loc['validation', 'end_date'][:10]} | {int(split_summary.loc['validation', 'home_count']):,} | ${split_summary.loc['validation', 'median_sale_price']:,.0f} | ${split_summary.loc['validation', 'median_price_per_sqft']:,.0f} |
| Test | {split_summary.loc['test', 'start_date'][:10]} | {split_summary.loc['test', 'end_date'][:10]} | {int(split_summary.loc['test', 'home_count']):,} | ${split_summary.loc['test', 'median_sale_price']:,.0f} | ${split_summary.loc['test', 'median_price_per_sqft']:,.0f} |

Vertical dashed lines in the time-series plots mark:

- Validation start: `{split_boundaries['validation_start'].date()}`
- Test start: `{split_boundaries['test_start'].date()}`

## Sales And Price Movement

![Monthly sales volume and prices](monthly_sales_volume_and_prices.png)

Major understanding: sale volume is seasonal, while price per square foot rises
into the test period. Median price per square foot moves from about
`${price_per_sqft_start:,.0f}` to `${price_per_sqft_end:,.0f}`, a
`{ppsf_change:.1%}` change over the dataset window.

![Monthly price per square foot](monthly_price_per_sqft.png)

## Macro Environment

![Macro environment over time](macro_environment_over_time.png)

Major understanding: the macro backdrop becomes more supportive for housing over
the sample. The 30-year mortgage rate moves from `{mortgage_start:.2f}%` to
`{mortgage_end:.2f}%`, while the Seattle Case-Shiller index changes by
`{case_shiller_change:.1%}`.

![Rates supply and starts](rates_supply_and_starts.png)

## Home Distribution And Split Mix

![Price distribution](price_distribution.png)

The raw price distribution is right-skewed, which explains why RMSE can look
large: luxury and unusual homes create large dollar errors.

![Split price distribution](split_price_distribution.png)

![Split property profile](split_property_profile.png)

![Top zipcode mix by split](top_zipcode_mix_by_split.png)

Major understanding: train, validation, and test periods are similar enough to
model, but not identical. The time-based test period has a stronger price-per-
square-foot environment and a potentially different geographic/property mix.
That helps explain why a static model trained on earlier sales can underpredict
later prices.

## Why This Matters For The Pricing Project

This EDA supports two modeling choices:

1. A time-based split is appropriate because pricing future homes from historical
   sales is the real operating problem.
2. Macro and sale-month features matter because the test period occurs in a
   changing price environment.

It also motivates the later feedback-loop idea: even a strong AVM can become
systematically biased when the market level changes between acquisition and
resale.
"""

    (EDA_OUTPUT_DIR / "eda_report.md").write_text(report)


def main() -> None:
    """Run King County housing and macro environment EDA."""

    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sales = load_sales_with_macro()
    sales = assign_model_splits(sales)
    sales["price_per_sqft"] = sales["price"] / sales["sqft_living"]

    monthly = summarize_monthly_sales(sales)
    split_boundaries = get_split_boundaries(sales)

    plot_sales_volume_and_prices(monthly, split_boundaries)
    plot_price_per_sqft(monthly, split_boundaries)
    plot_macro_environment(monthly, split_boundaries)
    plot_rates_and_supply(monthly, split_boundaries)
    plot_price_distribution(sales)
    plot_split_price_distribution(sales)
    plot_split_property_profile(sales)
    plot_top_zipcode_mix(sales)
    save_summary_tables(sales, monthly)
    write_eda_report(sales, monthly, split_boundaries)

    print("Saved EDA outputs to outputs/1_eda/")
    print(pd.read_csv(EDA_OUTPUT_DIR / "dataset_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
