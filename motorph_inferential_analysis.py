import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


BASE_DIR = Path(__file__).resolve().parent
SALES_FILENAME = "MotorPH_Sales_Cleaned_Q3_2025.csv"
DATA_CANDIDATES = [
    BASE_DIR / "MotorPH_Product&Sales" / SALES_FILENAME,
    BASE_DIR / "MotorPH_Data_Preprocessing" / SALES_FILENAME,
]
OUTPUT_DIR = BASE_DIR / "MotorPH_Inferential_Analysis_Results"
CHART_FILE = OUTPUT_DIR / "MotorPH_Q3_2025_Sales_Analysis.png"
SUMMARY_FILE = OUTPUT_DIR / "MotorPH_Q3_2025_Inferential_Results.txt"
REQUIRED_COLUMNS = {
    "date",
    "client_type",
    "product_name",
    "unit_price",
    "quantity",
    "total",
    "payment_method",
}
Q3_START = pd.Timestamp("2025-06-01")
Q3_END = pd.Timestamp("2025-08-31")
ALPHA = 0.05


def find_sales_file() -> Path:
    for path in DATA_CANDIDATES:
        if path.is_file():
            return path
    expected = "\n".join(f"- {path}" for path in DATA_CANDIDATES)
    raise FileNotFoundError(
        f"Could not find {SALES_FILENAME}. Checked these locations:\n{expected}"
    )


def load_and_validate_sales(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    sales = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(sales.columns)
    if missing_columns:
        raise ValueError(f"Sales CSV is missing columns: {sorted(missing_columns)}")
    if sales.empty:
        raise ValueError("The sales CSV has no rows.")

    original_rows = len(sales)
    duplicate_rows = int(sales.duplicated().sum())
    if duplicate_rows:
        raise ValueError(f"The cleaned sales CSV contains {duplicate_rows} duplicate rows.")

    text_columns = ["client_type", "product_name", "payment_method"]
    for column in text_columns:
        sales[column] = sales[column].astype("string").str.strip()
        sales[column] = sales[column].replace("", pd.NA)

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    numeric_columns = ["unit_price", "quantity", "total"]
    for column in numeric_columns:
        sales[column] = pd.to_numeric(sales[column], errors="coerce")

    missing_counts = sales[list(REQUIRED_COLUMNS)].isna().sum()
    if missing_counts.any():
        invalid = missing_counts[missing_counts.gt(0)].to_dict()
        raise ValueError(f"The cleaned sales CSV has missing or invalid values: {invalid}")

    for column in numeric_columns:
        if sales[column].le(0).any():
            raise ValueError(f"Sales values in {column!r} must be positive.")
        if sales[column].mod(1).ne(0).any():
            raise ValueError(f"Sales values in {column!r} must be whole numbers.")
        sales[column] = sales[column].astype("int64")

    incorrect_totals = sales["total"].ne(sales["unit_price"] * sales["quantity"])
    if incorrect_totals.any():
        raise ValueError(
            f"{int(incorrect_totals.sum())} sales totals do not equal "
            "unit_price multiplied by quantity."
        )

    q3_sales = sales.loc[sales["date"].between(Q3_START, Q3_END)].copy()
    if q3_sales.empty:
        raise ValueError("No sales records were found in Q3 2025.")

    quality = {
        "input_rows": original_rows,
        "duplicate_rows": duplicate_rows,
        "valid_rows": len(sales),
        "q3_rows": len(q3_sales),
        "q3_sales_days": int(q3_sales["date"].dt.normalize().nunique()),
    }
    return q3_sales, quality


def build_daily_sales(q3_sales: pd.DataFrame) -> pd.DataFrame:
    calendar_days = pd.date_range(Q3_START, Q3_END, freq="D")
    daily = (
        q3_sales.assign(date=q3_sales["date"].dt.normalize())
        .groupby("date")
        .agg(revenue=("total", "sum"), units=("quantity", "sum"))
        .reindex(calendar_days, fill_value=0)
    )
    daily.index.name = "date"
    daily["month"] = daily.index.to_period("M").astype(str)
    return daily


def kruskal_wallis(
    daily: pd.DataFrame, measure: str
) -> tuple[float, float, int]:
    ranked = daily[[measure, "month"]].copy()
    group_count = ranked["month"].nunique()
    if group_count != 3:
        raise ValueError(f"Expected three monthly groups for {measure}, found {group_count}.")
    ranked["rank"] = ranked[measure].rank(method="average")
    sample_size = len(ranked)
    rank_sum_terms = sum(
        group["rank"].sum() ** 2 / len(group)
        for _, group in ranked.groupby("month", sort=True)
    )
    statistic = (
        12 * rank_sum_terms / (sample_size * (sample_size + 1))
        - 3 * (sample_size + 1)
    )

    tie_counts = ranked[measure].value_counts().to_numpy()
    tie_correction = 1 - (
        (tie_counts**3 - tie_counts).sum()
        / (sample_size**3 - sample_size)
    )
    if tie_correction <= 0:
        raise ValueError(f"Cannot calculate a valid rank test for {measure}.")
    statistic = max(0.0, statistic / tie_correction)

    # A chi-square distribution with 2 degrees of freedom has survival function exp(-H/2).
    p_value = math.exp(-statistic / 2)
    return statistic, p_value, sample_size


def describe_daily_distributions(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, group in daily.groupby("month", sort=True):
        for measure, label in (("revenue", "Revenue (PHP)"), ("units", "Units")):
            values = group[measure]
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            rows.append(
                {
                    "month": month,
                    "measure": label,
                    "days": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(values.median()),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def build_chart(daily: pd.DataFrame, monthly: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    month_labels = [pd.Period(month).strftime("%b") for month in monthly.index]
    month_groups = [
        daily.loc[daily["month"].eq(month), "revenue"].to_numpy()
        for month in monthly.index
    ]
    unit_groups = [
        daily.loc[daily["month"].eq(month), "units"].to_numpy()
        for month in monthly.index
    ]

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    figure.suptitle("MotorPH Q3 2025 Sales Analysis", fontsize=16, fontweight="bold")

    axes[0, 0].bar(month_labels, monthly["revenue"], color="#258C8C")
    axes[0, 0].set_title("Monthly revenue")
    axes[0, 0].set_ylabel("Revenue (PHP millions)")
    axes[0, 0].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1_000_000:,.0f}")
    )
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(month_labels, monthly["units"], color="#E08A3A")
    axes[0, 1].set_title("Monthly units sold")
    axes[0, 1].set_ylabel("Units")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].boxplot(month_groups)
    axes[1, 0].set_xticks(range(1, len(month_labels) + 1), month_labels)
    axes[1, 0].set_title("Distribution of daily revenue")
    axes[1, 0].set_ylabel("Daily revenue (PHP)")
    axes[1, 0].ticklabel_format(axis="y", style="plain")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].boxplot(unit_groups)
    axes[1, 1].set_xticks(range(1, len(month_labels) + 1), month_labels)
    axes[1, 1].set_title("Distribution of daily units")
    axes[1, 1].set_ylabel("Daily units")
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(CHART_FILE, dpi=160, bbox_inches="tight")
    plt.close(figure)


def format_money(value: float) -> str:
    return f"PHP {value:,.2f}"


def build_summary(
    source: Path,
    q3_sales: pd.DataFrame,
    quality: dict[str, int],
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    distributions: pd.DataFrame,
    revenue_test: tuple[float, float, int],
    units_test: tuple[float, float, int],
) -> str:
    missing_days = (Q3_END - Q3_START).days + 1 - quality["q3_sales_days"]
    lines = [
        "MOTORPH Q3 2025 INFERENTIAL SALES ANALYSIS",
        "",
        f"Input file: {source}",
        f"Analysis period: {Q3_START:%B %d, %Y} to {Q3_END:%B %d, %Y}",
        "",
        "1. DATA VALIDATION",
        f"- Input rows: {quality['input_rows']:,}",
        f"- Exact duplicate rows: {quality['duplicate_rows']:,}",
        f"- Valid rows after checks: {quality['valid_rows']:,}",
        f"- Q3 transaction rows: {quality['q3_rows']:,}",
        f"- Q3 calendar days with sales records: {quality['q3_sales_days']:,} of 92",
        "- Required fields are complete; prices, quantities, and totals are positive whole numbers.",
        "- Every transaction total equals unit price multiplied by quantity.",
        "",
        "2. DESCRIPTIVE ANALYSIS",
        f"- Q3 revenue: {format_money(monthly['revenue'].sum())}",
        f"- Q3 units sold: {int(monthly['units'].sum()):,}",
        f"- Q3 sales rows: {quality['q3_rows']:,}",
        "- Transaction-level descriptive statistics:",
        "  Variable          Mean             Std dev          Median           Min              Max",
    ]
    for column, label in (
        ("unit_price", "Unit price"),
        ("quantity", "Quantity"),
        ("total", "Transaction total"),
    ):
        values = q3_sales[column]
        if column == "unit_price" or column == "total":
            lines.append(
                f"  {label:17} {format_money(values.mean()):>16} "
                f"{format_money(values.std()):>16} {format_money(values.median()):>16} "
                f"{format_money(values.min()):>16} {format_money(values.max()):>16}"
            )
        else:
            lines.append(
                f"  {label:17} {values.mean():>16,.2f} {values.std():>16,.2f} "
                f"{values.median():>16,.2f} {values.min():>16,.0f} {values.max():>16,.0f}"
            )
    lines.append("- Monthly sales:")

    for month, values in monthly.iterrows():
        lines.append(
            f"  {pd.Period(month).strftime('%B')}: "
            f"{format_money(values['revenue'])}; "
            f"{int(values['units']):,} units; "
            f"{int(values['sales_rows']):,} sales rows"
        )
    lines.append("- Month-to-month changes:")
    for previous_month, current_month in zip(monthly.index[:-1], monthly.index[1:]):
        previous = monthly.loc[previous_month]
        current = monthly.loc[current_month]
        revenue_change = (
            (current["revenue"] / previous["revenue"] - 1) * 100
            if previous["revenue"]
            else float("nan")
        )
        units_change = (
            (current["units"] / previous["units"] - 1) * 100
            if previous["units"]
            else float("nan")
        )
        lines.append(
            f"  {pd.Period(current_month).strftime('%B')} vs "
            f"{pd.Period(previous_month).strftime('%B')}: "
            f"revenue {revenue_change:+.2f}%; units {units_change:+.2f}%"
        )

    lines.extend(
        [
            "",
            "Daily distributions by month (mean, standard deviation, median, quartiles):",
            "Month    Measure       Days       Mean        Std dev       Median       Q1          Q3          IQR",
        ]
    )
    for row in distributions.itertuples(index=False):
        if row.measure == "Revenue (PHP)":
            lines.append(
                f"{pd.Period(row.month).strftime('%b'):6}  Revenue     {row.days:>3}  "
                f"{format_money(row.mean):>15}  {format_money(row.std):>15}  "
                f"{format_money(row.median):>15}  {format_money(row.q1):>15}  "
                f"{format_money(row.q3):>15}  {format_money(row.iqr):>15}"
            )
        else:
            lines.append(
                f"{pd.Period(row.month).strftime('%b'):6}  Units       {row.days:>3}  "
                f"{row.mean:>10.2f}  {row.std:>10.2f}  {row.median:>10.2f}  "
                f"{row.q1:>10.2f}  {row.q3:>10.2f}  {row.iqr:>10.2f}"
            )

    lines.extend(
        [
            "",
            "3. HYPOTHESIS TESTING",
            "Method: Kruskal-Wallis H test comparing daily totals across June, July, and August.",
            "Significance level: alpha = 0.05. P-values use the chi-square approximation (df = 2).",
            "",
            "Test A: Daily revenue",
            "H0: Daily revenue distributions are the same in June, July, and August.",
            "H1: At least one month's daily revenue distribution differs.",
            f"Result: H(2) = {revenue_test[0]:.3f}; p = {revenue_test[1]:.3f}; n = {revenue_test[2]}.",
            (
                "Decision: "
                + (
                    "Reject H0."
                    if revenue_test[1] < ALPHA
                    else "Fail to reject H0; p is not below alpha."
                )
            ),
            (
                "Interpretation: "
                + (
                    "There is evidence that daily revenue distributions differ across these months."
                    if revenue_test[1] < ALPHA
                    else "There is not enough evidence to conclude daily revenue distributions differ across these months."
                )
            ),
            "",
            "Test B: Daily units sold",
            "H0: Daily unit distributions are the same in June, July, and August.",
            "H1: At least one month's daily unit distribution differs.",
            f"Result: H(2) = {units_test[0]:.3f}; p = {units_test[1]:.3f}; n = {units_test[2]}.",
            (
                "Decision: "
                + (
                    "Reject H0."
                    if units_test[1] < ALPHA
                    else "Fail to reject H0; p is not below alpha."
                )
            ),
            (
                "Interpretation: "
                + (
                    "There is evidence that daily unit distributions differ across these months."
                    if units_test[1] < ALPHA
                    else "There is not enough evidence to conclude daily unit distributions differ across these months."
                )
            ),
            "",
            "4. INTERPRETATION AND LIMITATIONS",
            (
                f"Daily totals are available for {quality['q3_sales_days']} of 92 Q3 dates; "
                f"{missing_days} dates have no transaction row."
            ),
            "Kruskal-Wallis is an omnibus rank-based test. It does not identify which month pairs differ.",
            "Failing to reject H0 does not prove that the months are identical.",
            "Daily observations may be time-dependent, so the independence assumption may not hold.",
            "These results describe one quarter and do not establish a causal effect or predict future sales.",
            "Check the saved chart for monthly totals and daily distribution boxplots.",
            f"Chart: {CHART_FILE}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    source = find_sales_file()
    q3_sales, quality = load_and_validate_sales(source)
    daily = build_daily_sales(q3_sales)
    monthly = (
        q3_sales.assign(month=q3_sales["date"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(
            revenue=("total", "sum"),
            units=("quantity", "sum"),
            sales_rows=("date", "size"),
        )
        .reindex(["2025-06", "2025-07", "2025-08"], fill_value=0)
    )
    distributions = describe_daily_distributions(daily)
    revenue_test = kruskal_wallis(daily, "revenue")
    units_test = kruskal_wallis(daily, "units")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_chart(daily, monthly)
    summary = build_summary(
        source,
        q3_sales,
        quality,
        daily,
        monthly,
        distributions,
        revenue_test,
        units_test,
    )
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Results saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
