from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter, MaxNLocator
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "MotorPH_Product&Sales"
PRODUCTS_FILE = DATA_DIR / "MotorPH_Products_Cleaned_2025.csv"
SALES_FILE = DATA_DIR / "MotorPH_Sales_Cleaned_Q3_2025.csv"
PRICE_HISTORY_FILE = DATA_DIR / "MotorPH_Price_History_2025.csv"
PRICE_REVIEW_FILE = DATA_DIR / "MotorPH_Price_Variance_Review.csv"
DASHBOARD_FILE = BASE_DIR / "MotorPH_Analytical_Dashboard.png"
INTERACTIVE_DASHBOARD_FILE = BASE_DIR / "MotorPH_Interactive_Dashboard.html"

PRODUCT_COLUMNS = {
    "Product Name",
    "Product Type",
    "Unit Price",
}
SALES_COLUMNS = {
    "date",
    "client_type",
    "product_name",
    "unit_price",
    "quantity",
    "total",
    "payment_method",
}
SMALL_CATEGORY_LABEL = "Other (<5%)"
PRICE_HISTORY_COLUMNS = {"product_name", "unit_price", "valid_from", "valid_to"}


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {sorted(missing)}")


def validate_price_variances(sales: pd.DataFrame) -> pd.DataFrame:
    current_price_mismatch = sales["unit_price"].ne(sales["catalog_unit_price"])
    sales["price_validation_status"] = "Matches current catalog"
    review_columns = [
        "date",
        "product_name",
        "unit_price",
        "catalog_unit_price",
        "quantity",
        "total",
    ]
    review = sales.loc[current_price_mismatch, review_columns].copy()
    review["validation_status"] = "Unverified: effective-dated price history required"

    if not review.empty and PRICE_HISTORY_FILE.is_file():
        history = pd.read_csv(PRICE_HISTORY_FILE)
        require_columns(history, PRICE_HISTORY_COLUMNS, PRICE_HISTORY_FILE)
        history["product_name"] = (
            history["product_name"].astype("string").str.strip().replace("", pd.NA)
        )
        history["unit_price"] = pd.to_numeric(history["unit_price"], errors="coerce")
        raw_valid_to = history["valid_to"].astype("string").str.strip()
        history["valid_from"] = pd.to_datetime(history["valid_from"], errors="coerce")
        history["valid_to"] = pd.to_datetime(history["valid_to"], errors="coerce")
        if (
            history[["product_name", "unit_price", "valid_from"]].isna().any().any()
            or (
                raw_valid_to.notna()
                & raw_valid_to.ne("")
                & history["valid_to"].isna()
            ).any()
            or history["unit_price"].le(0).any()
            or history["unit_price"].mod(1).ne(0).any()
            or history["valid_to"].lt(history["valid_from"]).any()
        ):
            raise ValueError(
                "Price history must have product names, positive whole-number prices, "
                "valid start dates, and end dates on or after start dates."
            )
        history["unit_price"] = history["unit_price"].astype("int64")
        history = history.sort_values(["product_name", "valid_from"])
        for product_name, periods in history.groupby("product_name", sort=False):
            previous_end = pd.NaT
            for position, period in enumerate(periods.itertuples(index=False)):
                if position and (
                    pd.isna(previous_end)
                    or period.valid_from <= previous_end
                ):
                    raise ValueError(
                        f"Overlapping price-history periods for {product_name}."
                    )
                previous_end = period.valid_to

        statuses = []
        for row in review.itertuples():
            effective_prices = history.loc[
                history["product_name"].eq(row.product_name)
                & history["valid_from"].le(row.date)
                & (history["valid_to"].isna() | history["valid_to"].ge(row.date)),
                "unit_price",
            ]
            if len(effective_prices) == 1 and int(effective_prices.iloc[0]) == row.unit_price:
                statuses.append("Verified by effective-dated price history")
            else:
                statuses.append(
                    "Unverified: no matching effective-dated price record"
                )
        review["validation_status"] = statuses

    if not review.empty:
        sales.loc[
            current_price_mismatch, "price_validation_status"
        ] = review["validation_status"].to_numpy()
    review.to_csv(PRICE_REVIEW_FILE, index=False)
    if review["validation_status"].str.startswith("Unverified").any():
        raise ValueError(
            f"{int(review['validation_status'].str.startswith('Unverified').sum())} "
            "sales rows differ from the current catalog and could not be verified "
            "against effective-dated price history. Review "
            f"{PRICE_REVIEW_FILE} before generating the dashboard."
        )
    return sales


def build_key_findings(
    products: pd.DataFrame, sales: pd.DataFrame
) -> list[str]:
    product_categories = products.set_index("Product Name")["Product Type"]
    product_sales = sales.groupby("product_name").agg(
        revenue=("total", "sum"),
        units=("quantity", "sum"),
    )
    categories = sales["product_name"].map(product_categories)
    dashboard_categories = collapse_small_categories(products, categories)
    category_revenue = sales.groupby(dashboard_categories)["total"].sum()
    total_revenue = int(sales["total"].sum())

    latest_month = sales["date"].max().to_period("M")
    quarter_start = (latest_month - 2).start_time
    quarter_end = latest_month.end_time
    quarter_sales = sales.loc[sales["date"].between(quarter_start, quarter_end)]
    quarter_revenue = int(quarter_sales["total"].sum())
    quarter_share = quarter_revenue / total_revenue * 100 if total_revenue else 0

    other_mask = dashboard_categories.eq(SMALL_CATEGORY_LABEL)
    heritage_sales = sales.loc[other_mask & categories.eq("Heritage")]
    heritage_revenue = int(
        heritage_sales["total"].sum()
    )
    heritage_products = (
        heritage_sales.groupby("product_name")["total"]
        .sum()
        .sort_values(ascending=False)
    )
    heritage_examples = ", ".join(heritage_products.head(2).index)
    other_revenue = int(category_revenue.get(SMALL_CATEGORY_LABEL, 0))
    heritage_share = heritage_revenue / other_revenue * 100 if other_revenue else 0

    top_units = product_sales["units"].idxmax()
    top_revenue = product_sales["revenue"].idxmax()
    unknown_clients = int(sales["client_type"].str.casefold().eq("unknown").sum())
    unknown_payments = int(sales["payment_method"].str.casefold().eq("unknown").sum())
    price_review_rows = int(
        sales["price_validation_status"].str.startswith("Verified").sum()
    )
    current_price_matches = int(
        sales["price_validation_status"].eq("Matches current catalog").sum()
    )

    return [
        (
            f"{quarter_start:%b}-{quarter_end:%b %Y} contributed "
            f"{money(quarter_revenue)} ({quarter_share:.2f}%) of recorded revenue. "
            "The quarter's monthly totals are close, indicating sustained sales rather "
            "than a single-month spike."
        ),
        (
            f"{SMALL_CATEGORY_LABEL} is the largest revenue bucket at "
            f"{money(other_revenue)} ({other_revenue / total_revenue * 100:.1f}%). "
            f"Heritage models contribute {money(heritage_revenue)} "
            f"({heritage_share:.1f}% of that bucket), led by {heritage_examples}. "
            "Grouping small categories hides meaningful premium-product revenue."
        ),
        (
            f"{top_units} leads volume ({int(product_sales.loc[top_units, 'units']):,} units), "
            f"while {top_revenue} leads revenue "
            f"({money(int(product_sales.loc[top_revenue, 'revenue']))}). "
            "Unit demand and revenue therefore point to different products to prioritize."
        ),
        (
            f"Data review: {unknown_clients} unknown client labels and {unknown_payments} "
            f"unknown payment labels are retained. {current_price_matches} transaction "
            f"prices match the current catalog; {price_review_rows} differences were "
            "verified against dated price history."
        ),
    ]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (PRODUCTS_FILE, SALES_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required MotorPH dataset not found: {path}")

    products = pd.read_csv(PRODUCTS_FILE)
    sales = pd.read_csv(SALES_FILE)
    require_columns(products, PRODUCT_COLUMNS, PRODUCTS_FILE)
    require_columns(sales, SALES_COLUMNS, SALES_FILE)

    product_duplicates = int(products.duplicated().sum())
    products = products.drop_duplicates().copy()
    if product_duplicates:
        print(f"Removed {product_duplicates} exact duplicate product rows.")

    for column in ("Product Name", "Product Type"):
        products[column] = products[column].astype("string").str.strip().replace("", pd.NA)
    if products.empty or products[list(PRODUCT_COLUMNS)].isna().any().any():
        raise ValueError("The product catalog is empty or has missing names/categories/prices.")
    if products["Product Name"].duplicated().any():
        raise ValueError("Duplicate product names remain in the product catalog.")

    products["Unit Price"] = pd.to_numeric(products["Unit Price"], errors="coerce")
    if products["Unit Price"].isna().any() or products["Unit Price"].le(0).any():
        raise ValueError("Product unit prices must be present, numeric, and greater than zero.")
    if products["Unit Price"].mod(1).ne(0).any():
        raise ValueError("Product unit prices must be whole-number amounts.")
    products["Unit Price"] = products["Unit Price"].astype("int64")

    sales_duplicates = int(sales.duplicated().sum())
    sales = sales.drop_duplicates().copy()
    if sales_duplicates:
        print(f"Removed {sales_duplicates} exact duplicate sales rows.")

    for column in ("client_type", "product_name", "payment_method"):
        sales[column] = (
            sales[column].astype("string").str.strip().replace("", pd.NA)
        )
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    numeric_columns = ("unit_price", "quantity", "total")
    for column in numeric_columns:
        sales[column] = pd.to_numeric(sales[column], errors="coerce")
    numeric_sales = sales[list(numeric_columns)].dropna()
    if numeric_sales.mod(1).ne(0).any().any():
        raise ValueError("Sales prices, quantities, and totals must be whole-number amounts.")

    required_sales_values = [
        "date",
        "client_type",
        "product_name",
        "unit_price",
        "quantity",
        "total",
        "payment_method",
    ]
    invalid_counts = sales[required_sales_values].isna().sum()
    rows_to_remove = int(sales[required_sales_values].isna().any(axis=1).sum())
    if rows_to_remove:
        print(
            "Sales rows with missing or invalid required values by field: "
            f"{invalid_counts[invalid_counts.gt(0)].to_dict()}"
        )
        sales = sales.dropna(subset=required_sales_values).copy()
        print(f"Removed {rows_to_remove} sales rows with invalid/missing required values.")
    if sales.empty:
        raise ValueError("No valid sales rows remain after cleaning.")
    if sales[["unit_price", "quantity", "total"]].le(0).any().any():
        raise ValueError("Sales prices, quantities, and totals must all be greater than zero.")

    catalog = products[["Product Name", "Product Type", "Unit Price"]].rename(
        columns={
            "Product Name": "product_name",
            "Product Type": "product_category",
            "Unit Price": "catalog_unit_price",
        }
    )
    sales = sales.merge(
        catalog,
        on="product_name",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = sales.loc[sales["_merge"].ne("both"), "product_name"].drop_duplicates()
    if not unmatched.empty:
        raise ValueError(f"Sales products missing from catalog: {unmatched.tolist()}")
    sales = sales.drop(columns="_merge")

    total_mismatches = sales["total"].ne(sales["unit_price"] * sales["quantity"])
    if total_mismatches.any():
        raise ValueError(
            f"{int(total_mismatches.sum())} sales totals do not equal unit price x quantity."
        )

    sales["unit_price"] = sales["unit_price"].astype("int64")
    sales["quantity"] = sales["quantity"].astype("int64")
    sales["total"] = sales["total"].astype("int64")
    sales = validate_price_variances(sales)
    return products, sales


def collapse_small_categories(
    products: pd.DataFrame, categories: pd.Series
) -> pd.Series:
    shares = products["Product Type"].value_counts(normalize=True)
    small_categories = shares[shares < 0.05].index
    return categories.where(~categories.isin(small_categories), SMALL_CATEGORY_LABEL)


def money(value: float) -> str:
    return f"PHP {value:,.0f}"


def build_interactive_dashboard(products: pd.DataFrame, sales: pd.DataFrame) -> None:
    sales = sales.copy()
    key_findings = build_key_findings(products, sales)
    key_findings.append(
        "Price control: totals must equal transaction price x quantity. Catalog price "
        "differences require effective-dated history; unverified rows are written to "
        f"{PRICE_REVIEW_FILE.name} and stop dashboard generation."
    )
    sales["dashboard_category"] = collapse_small_categories(
        products, sales["product_category"]
    )
    sales["month"] = sales["date"].dt.to_period("M").dt.to_timestamp()
    monthly = sales.groupby("month").agg(
        revenue=("total", "sum"),
        units=("quantity", "sum"),
    )
    complete_months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(complete_months.to_timestamp(), fill_value=0)
    product_sales = sales.groupby("product_name").agg(
        revenue=("total", "sum"),
        units=("quantity", "sum"),
    )
    category_revenue = (
        sales.groupby("dashboard_category")["total"].sum().sort_values()
    )
    category_counts = collapse_small_categories(
        products, products["Product Type"]
    ).value_counts().sort_values()

    inventory = products[
        ["Product Name", "Product Type", "Unit Price"]
    ].rename(
        columns={
            "Product Name": "Product",
            "Product Type": "Category",
            "Unit Price": "Current catalog price (PHP)",
        }
    )
    inventory["Units sold"] = inventory["Product"].map(product_sales["units"]).fillna(0)
    inventory["Revenue (PHP)"] = (
        inventory["Product"].map(product_sales["revenue"]).fillna(0)
    )
    inventory = inventory.sort_values(["Category", "Product"])
    inventory["Units sold"] = inventory["Units sold"].astype("int64")
    inventory["Current catalog price (PHP)"] = inventory[
        "Current catalog price (PHP)"
    ].map(
        lambda value: money(value)
    )
    inventory["Revenue (PHP)"] = inventory["Revenue (PHP)"].map(money)

    start_date = sales["date"].min().strftime("%b %d, %Y")
    end_date = sales["date"].max().strftime("%b %d, %Y")
    fig = make_subplots(
        rows=5,
        cols=2,
        specs=[
            [{"type": "table", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "table", "colspan": 2}, None],
        ],
        subplot_titles=[
            "Key findings",
            "Sales Performance | Monthly revenue",
            "Sales Performance | Monthly units sold",
            "Product Performance | Top 10 by revenue",
            "Product Performance | Top 10 by units sold",
            "Sales by Product Category | Catalog counts",
            "Sales by Product Category | Revenue",
            "Inventory | Catalog, pricing & sales",
        ],
        row_heights=[0.16, 0.17, 0.20, 0.17, 0.30],
        vertical_spacing=0.06,
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Table(
            header={
                "values": ["Key finding"],
                "fill_color": "#17324D",
                "font": {"color": "white", "size": 12},
                "align": "left",
                "height": 28,
            },
            cells={
                "values": [key_findings],
                "fill_color": [
                    ["white" if row % 2 == 0 else "#F0F4F8" for row in range(len(key_findings))]
                ],
                "font": {"color": "#263746", "size": 11},
                "align": "left",
                "height": 34,
            },
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["revenue"],
            mode="lines+markers",
            name="Revenue",
            line={"color": "#258C8C", "width": 3},
            marker={"size": 8},
            hovertemplate="%{x|%b %Y}<br>Revenue: PHP %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["units"],
            mode="lines+markers",
            name="Units sold",
            line={"color": "#E08A3A", "width": 3},
            marker={"size": 8},
            hovertemplate="%{x|%b %Y}<br>Units: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    top_revenue = product_sales.nlargest(10, "revenue").sort_values("revenue")
    fig.add_trace(
        go.Bar(
            x=top_revenue["revenue"],
            y=top_revenue.index,
            orientation="h",
            name="Revenue",
            marker_color="#258C8C",
            hovertemplate="%{y}<br>Revenue: PHP %{x:,.0f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    top_units = product_sales.nlargest(10, "units").sort_values("units")
    fig.add_trace(
        go.Bar(
            x=top_units["units"],
            y=top_units.index,
            orientation="h",
            name="Units sold",
            marker_color="#E08A3A",
            hovertemplate="%{y}<br>Units sold: %{x:,.0f}<extra></extra>",
        ),
        row=3,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation="h",
            name="Products",
            marker_color="#527DA8",
            hovertemplate="%{y}<br>Products: %{x}<extra></extra>",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=category_revenue.values,
            y=category_revenue.index,
            orientation="h",
            name="Category revenue",
            marker_color="#7656A5",
            hovertemplate="%{y}<br>Revenue: PHP %{x:,.0f}<extra></extra>",
        ),
        row=4,
        col=2,
    )
    fig.add_trace(
        go.Table(
            header={
                "values": inventory.columns.tolist(),
                "fill_color": "#17324D",
                "font": {"color": "white", "size": 12},
                "align": "left",
                "height": 30,
            },
            cells={
                "values": [inventory[column] for column in inventory.columns],
                "fill_color": [
                    ["white" if row % 2 == 0 else "#F0F4F8" for row in range(len(inventory))]
                ] * len(inventory.columns),
                "font": {"color": "#263746", "size": 11},
                "align": "left",
                "height": 24,
            },
        ),
        row=5,
        col=1,
    )

    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Revenue (PHP)", tickformat=".2s", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=2)
    fig.update_yaxes(title_text="Units", rangemode="tozero", row=2, col=2)
    fig.update_xaxes(title_text="Revenue (PHP)", tickformat=".2s", row=3, col=1)
    fig.update_xaxes(title_text="Units sold", row=3, col=2)
    fig.update_xaxes(title_text="Products", row=4, col=1)
    fig.update_xaxes(title_text="Revenue (PHP)", tickformat=".2s", row=4, col=2)
    fig.update_yaxes(autorange="reversed", row=3, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=2)
    fig.update_yaxes(autorange="reversed", row=4, col=1)
    fig.update_yaxes(autorange="reversed", row=4, col=2)

    total_revenue = int(sales["total"].sum())
    total_units = int(sales["quantity"].sum())
    metric_cards = [
        ("TOTAL REVENUE", money(total_revenue)),
        ("UNITS SOLD", f"{total_units:,}"),
        ("SALES ROWS", f"{len(sales):,}"),
        ("PRODUCTS", f"{len(products):,}"),
        ("CATEGORIES", f"{products['Product Type'].nunique():,}"),
    ]
    for index, (label, value) in enumerate(metric_cards):
        fig.add_annotation(
            x=0.01 + index * 0.198,
            y=1.06,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="middle",
            text=f"<b>{label}</b><br><span style='font-size:20px'>{value}</span>",
            showarrow=False,
            align="left",
            bgcolor="white",
            bordercolor="#DCE4EC",
            borderwidth=1,
            borderpad=10,
            font={"color": "#17324D", "size": 11},
        )

    fig.update_layout(
        title={
            "text": (
                "MotorPH Overview | Products &amp; Sales Dashboard"
                f"<br><sup>Sales period: {start_date} – {end_date} | "
                "Currency: PHP | Categories under 5% grouped as Other</sup>"
            ),
            "x": 0.01,
            "xanchor": "left",
            "y": 0.99,
        },
        height=2200,
        template="plotly_white",
        paper_bgcolor="#F4F7FA",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#17324D"},
        margin={"l": 150, "r": 55, "t": 300, "b": 65},
        showlegend=False,
        hoverlabel={"bgcolor": "white", "font_size": 13},
    )
    fig.write_html(
        INTERACTIVE_DASHBOARD_FILE,
        full_html=True,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
        auto_open=False,
    )
    print(f"Interactive dashboard saved to: {INTERACTIVE_DASHBOARD_FILE}")


def build_dashboard(products: pd.DataFrame, sales: pd.DataFrame) -> None:
    sales = sales.copy()
    key_findings = build_key_findings(products, sales)
    sales["dashboard_category"] = collapse_small_categories(
        products, sales["product_category"]
    )
    sales["month"] = sales["date"].dt.to_period("M").dt.to_timestamp()

    monthly = sales.groupby("month").agg(
        revenue=("total", "sum"),
        units=("quantity", "sum"),
    )
    complete_months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(complete_months.to_timestamp(), fill_value=0)
    product_sales = sales.groupby("product_name").agg(
        revenue=("total", "sum"),
        units=("quantity", "sum"),
    )
    category_revenue = sales.groupby("dashboard_category")["total"].sum().sort_values()
    category_counts = collapse_small_categories(
        products, products["Product Type"]
    ).value_counts().sort_values()

    inventory = products[
        ["Product Name", "Product Type", "Unit Price"]
    ].rename(
        columns={
            "Product Name": "Product",
            "Product Type": "Category",
            "Unit Price": "Current catalog price",
        }
    )
    inventory["Units sold"] = inventory["Product"].map(product_sales["units"]).fillna(0)
    inventory["Revenue"] = inventory["Product"].map(product_sales["revenue"]).fillna(0)
    inventory = inventory.sort_values(["Category", "Product"])
    inventory["Current catalog price"] = inventory["Current catalog price"].map(money)
    inventory["Revenue"] = inventory["Revenue"].map(money)
    inventory["Units sold"] = inventory["Units sold"].astype("int64").map(
        lambda value: f"{value:,}"
    )

    start_date = sales["date"].min().strftime("%b %d, %Y")
    end_date = sales["date"].max().strftime("%b %d, %Y")
    monthly_labels = monthly.index.strftime("%b %Y")
    unknown_clients = int(sales["client_type"].str.casefold().eq("unknown").sum())
    unknown_payments = int(sales["payment_method"].str.casefold().eq("unknown").sum())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 26), facecolor="#F4F7FA")
    grid = fig.add_gridspec(
        8,
        1,
        height_ratios=[0.8, 1.15, 1.3, 3.2, 3.2, 3.6, 0.45, 11.6],
        hspace=0.36,
    )

    title_ax = fig.add_subplot(grid[0])
    title_ax.axis("off")
    title_ax.text(
        0.01,
        0.76,
        "MotorPH Overview | Products & Sales Dashboard",
        fontsize=24,
        fontweight="bold",
        color="#17324D",
        transform=title_ax.transAxes,
    )
    title_ax.text(
        0.01,
        0.22,
        f"Sales period: {start_date} – {end_date}  |  Currency: PHP  |  "
        "Source: supplied cleaned datasets",
        fontsize=10,
        color="#536579",
        transform=title_ax.transAxes,
    )

    cards_grid = grid[1].subgridspec(1, 5, wspace=0.12)
    metrics = [
        ("TOTAL REVENUE", money(sales["total"].sum())),
        ("UNITS SOLD", f"{sales['quantity'].sum():,}"),
        ("SALES ROWS", f"{len(sales):,}"),
        ("PRODUCTS", f"{len(products):,}"),
        ("CATEGORIES", f"{products['Product Type'].nunique():,}"),
    ]
    for index, (label, value) in enumerate(metrics):
        ax = fig.add_subplot(cards_grid[0, index])
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#DCE4EC")
        ax.text(0.08, 0.68, label, fontsize=8, color="#66788A", transform=ax.transAxes)
        ax.text(
            0.08,
            0.28,
            value,
            fontsize=15 if index != 0 else 13,
            fontweight="bold",
            color="#17324D",
            transform=ax.transAxes,
        )

    insight_ax = fig.add_subplot(grid[2])
    insight_ax.axis("off")
    insight_ax.text(
        0.01,
        0.98,
        "KEY FINDINGS",
        fontsize=10,
        fontweight="bold",
        color="#17324D",
        transform=insight_ax.transAxes,
        va="top",
    )
    for column, finding_group in enumerate((key_findings[:2], key_findings[2:])):
        insight_ax.text(
            0.01 + column * 0.50,
            0.79,
            "\n".join(
                fill(
                    finding,
                    width=72,
                    initial_indent="- ",
                    subsequent_indent="  ",
                )
                for finding in finding_group
            ),
            fontsize=8,
            color="#536579",
            transform=insight_ax.transAxes,
            va="top",
            wrap=True,
        )

    revenue_ax, units_ax = grid[3, 0].subgridspec(1, 2, wspace=0.20).subplots().flatten()
    revenue_ax.set_title(
        "Sales Performance | Monthly revenue",
        loc="left",
        fontweight="bold",
        color="#17324D",
    )
    revenue_ax.bar(monthly_labels, monthly["revenue"] / 1_000_000, color="#258C8C")
    revenue_ax.set_ylabel("Revenue (PHP millions)")
    revenue_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    revenue_ax.tick_params(axis="x", rotation=30)
    revenue_ax.grid(axis="x", visible=False)
    revenue_ax.set_axisbelow(True)

    units_ax.set_title(
        "Sales Performance | Monthly units sold",
        loc="left",
        fontweight="bold",
        color="#17324D",
    )
    units_ax.plot(
        monthly_labels,
        monthly["units"],
        color="#E08A3A",
        linewidth=2.5,
        marker="o",
        markersize=5,
    )
    units_ax.set_ylabel("Units")
    units_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    units_ax.tick_params(axis="x", rotation=30)
    units_ax.grid(axis="x", visible=False)
    units_ax.set_axisbelow(True)

    product_revenue_ax, product_units_ax = (
        grid[4, 0].subgridspec(1, 2, wspace=0.24).subplots().flatten()
    )
    top_revenue = product_sales.nlargest(10, "revenue").sort_values("revenue")
    bars = product_revenue_ax.barh(
        top_revenue.index, top_revenue["revenue"] / 1_000_000, color="#258C8C"
    )
    product_revenue_ax.bar_label(
        bars,
        labels=[f"{value:,.1f}" for value in top_revenue["revenue"] / 1_000_000],
        padding=3,
        fontsize=7,
    )
    product_revenue_ax.set_title(
        "Product Performance | Top 10 by revenue (PHP millions)",
        loc="left",
        fontweight="bold",
        color="#17324D",
    )
    product_revenue_ax.set_xlabel("Revenue (PHP millions)")
    product_revenue_ax.grid(axis="y", visible=False)
    product_revenue_ax.set_xlim(0, top_revenue["revenue"].max() / 1_000_000 * 1.22)

    top_units = product_sales.nlargest(10, "units").sort_values("units")
    bars = product_units_ax.barh(top_units.index, top_units["units"], color="#E08A3A")
    product_units_ax.bar_label(bars, padding=3, fontsize=7, fmt="%.0f")
    product_units_ax.set_title(
        "Product Performance | Top 10 by units sold",
        loc="left",
        fontweight="bold",
        color="#17324D",
    )
    product_units_ax.set_xlabel("Units sold")
    product_units_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    product_units_ax.grid(axis="y", visible=False)
    product_units_ax.set_xlim(0, top_units["units"].max() * 1.18)

    category_count_ax, category_revenue_ax = (
        grid[5, 0].subgridspec(1, 2, wspace=0.24).subplots().flatten()
    )
    bars = category_count_ax.barh(
        category_counts.index, category_counts.values, color="#527DA8"
    )
    category_count_ax.bar_label(bars, padding=3, fontsize=8, fmt="%.0f")
    category_count_ax.set_title(
        "Sales by Product Category | Catalog counts",
        loc="left",
        fontweight="bold",
        color="#17324D",
    )
    category_count_ax.set_xlabel("Products")
    category_count_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    category_count_ax.set_xlim(0, category_counts.max() * 1.22)
    category_count_ax.grid(axis="y", visible=False)

    revenue_values = category_revenue / 1_000_000
    bars = category_revenue_ax.barh(
        revenue_values.index, revenue_values.values, color="#7656A5"
    )
    category_revenue_ax.bar_label(
        bars, padding=3, fontsize=8, labels=[f"{value:,.1f}" for value in revenue_values]
    )
    category_revenue_ax.set_title(
        "Sales by Product Category | Revenue",
        loc="left",
        fontweight="bold",
        color="#17324D",
    )
    category_revenue_ax.set_xlabel("Revenue (PHP millions)")
    category_revenue_ax.set_xlim(0, revenue_values.max() * 1.25)
    category_revenue_ax.grid(axis="y", visible=False)

    inventory_title = fig.add_subplot(grid[6])
    inventory_title.axis("off")
    inventory_title.text(
        0.01,
        0.15,
        "Inventory | Catalog, pricing & sales",
        fontsize=13,
        fontweight="bold",
        color="#17324D",
        transform=inventory_title.transAxes,
    )
    table_ax = fig.add_subplot(grid[7])
    table_ax.axis("off")
    table = table_ax.table(
        cellText=inventory.values.tolist(),
        colLabels=inventory.columns.tolist(),
        cellLoc="left",
        colLoc="left",
        colWidths=[0.28, 0.18, 0.17, 0.14, 0.23],
        bbox=[0.01, 0.02, 0.98, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.25)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#DCE4EC")
        if row == 0:
            cell.set_facecolor("#17324D")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white" if row % 2 else "#F0F4F8")
            cell.get_text().set_color("#263746")

    fig.text(
        0.02,
        0.006,
        "Categories with less than 5% of the catalog are grouped as Other. "
        f"Unknown client labels: {unknown_clients:,}; unknown payment labels: {unknown_payments:,}. "
        "These source values are retained, not imputed.",
        fontsize=8,
        color="#536579",
    )
    fig.savefig(DASHBOARD_FILE, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Products: {len(products):,}; categories: {products['Product Type'].nunique():,}")
    print(f"Prepared sales rows: {len(sales):,}; exact duplicate rows remaining: {sales.duplicated().sum():,}")
    print(f"Sales date range: {start_date} to {end_date}")
    print(f"Missing values after preparation: {int(sales.isna().sum().sum()):,}")
    print(f"Unknown client labels: {unknown_clients:,}; unknown payment labels: {unknown_payments:,}")
    print(f"Total units sold: {sales['quantity'].sum():,}")
    print(f"Total revenue: {money(sales['total'].sum())}")
    print(f"Dashboard saved to: {DASHBOARD_FILE}")


if __name__ == "__main__":
    product_data, sales_data = load_data()
    build_dashboard(product_data, sales_data)
    build_interactive_dashboard(product_data, sales_data)
