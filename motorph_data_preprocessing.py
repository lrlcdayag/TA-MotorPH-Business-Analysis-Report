from pathlib import Path
import pandas as pd

SOURCE_DIR = Path(r"C:\python")

PRODUCTS_FILE = "MotorPH_Products_List_2025.csv"
SALES_FILE = "MotorPH_Sales Data-3rd Quarter-Year 2025.csv"


if not (SOURCE_DIR / PRODUCTS_FILE).is_file():
    raise FileNotFoundError(f"Could not find {PRODUCTS_FILE} in {SOURCE_DIR}")

OUTPUT_DIR = SOURCE_DIR / 'MotorPH_Data_Preprocessing'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Loads raw datasets
products_raw = pd.read_csv(SOURCE_DIR / PRODUCTS_FILE)
sales_raw = pd.read_csv(SOURCE_DIR / SALES_FILE)

print(f'Products: {products_raw.shape[0]:,} rows, {products_raw.shape[1]} columns')
print(f'Sales: {sales_raw.shape[0]:,} rows, {sales_raw.shape[1]} columns')
print("\nProduct Catalog Head:")
print(products_raw.head())
print("\nSales Data Head:")
print(sales_raw.head())


def quality_report(frame):
    return pd.DataFrame({
        'dtype': frame.dtypes.astype(str),
        'missing': frame.isna().sum(),
        'distinct': frame.nunique(dropna=True),
    })


print('\nProduct catalog quality:')
print(quality_report(products_raw))
print('Exact duplicate product rows:', products_raw.duplicated().sum())

print('\nSales quality:')
print(quality_report(sales_raw))
print('Exact duplicate sales rows:', sales_raw.duplicated().sum())

# Audit sales dates
sales_dates = pd.to_datetime(sales_raw['date'], errors='coerce', format='mixed')
print('\nMissing or unparseable sales dates:', int(sales_dates.isna().sum()))
print(sales_raw.loc[sales_dates.isna(), ['date', 'product']].value_counts(dropna=False).rename('rows').reset_index())

# Product name corrections dictionary
name_fixes = {
    'Bajaj CT12x': 'Bajaj CT125',
    'Benelli 502x': 'Benelli 502C',
    'Bristol Bobber 65x': 'Bristol Bobber 650',
    'CFMoto 300Sx': 'CFMoto 300SR',
    'Honda ADV 16x': 'Honda ADV 160',
    'KTM 790 Dukx': 'KTM 790 Duke',
    'Kawasaki KLX 23x': 'Kawasaki KLX 230',
    'Motorstar Xplorer 250x': 'Motorstar Xplorer 250R',
    'Suzuki Raider R150 Fx': 'Suzuki Raider R150 Fi',
    'TVS Apache RTR 200 4x': 'TVS Apache RTR 200 4V',
    'Yamaha MT-1x': 'Yamaha MT-15',
    'Yamaha Serow 25x': 'Yamaha Serow 250',
    'Yamaha Sniper 15x': 'Yamaha Sniper 155',
}

corrected_names = sales_raw['product'].astype('string').str.strip().replace(name_fixes)
catalog_price = products_raw.set_index('EntrName')['UnitPrice']
expected_price = corrected_names.map(catalog_price)

print('\nProduct names corrected:', int(corrected_names.ne(sales_raw['product']).sum()))
print('Products without a catalog match after correction:', int(expected_price.isna().sum()))
price_mismatch = expected_price.notna() & sales_raw['unitprice'].ne(expected_price)
print('Sales rows with a unit price different from the catalog:', int(price_mismatch.sum()))
print('Rows where source total differs from catalog price x quantity:', int((sales_raw['total'] != expected_price * sales_raw['quantity']).sum()))

# --- PREPROCESS PRODUCTS CATALOG ---
products_source = products_raw.drop_duplicates().copy()
if products_source['EntrName'].duplicated().any():
    raise ValueError('Duplicate product names need review before creating the report.')

product_names = products_source['EntrName'].astype('string').str.strip()
product_types = products_source['EntrDetails'].astype('string').str.split('/', n=1).str[0].str.strip()

products_clean = pd.DataFrame({
    'Product ID Number': range(1, len(products_source) + 1),
    'Product Name': product_names,
    'Product Type': product_types,
    'Unit Price': pd.to_numeric(products_source['UnitPrice'], errors='raise').astype('int64'),
    'Date of Manufacturing': pd.to_numeric(products_source['Manufacturing Date'], errors='raise').astype('int64'),
    'Date of Acquisition': pd.to_numeric(products_source['Acquisiton'], errors='raise').astype('int64'),
})

products_output = OUTPUT_DIR / 'MotorPH_Products_Cleaned_2025.csv'
products_clean.to_csv(products_output, index=False)
print(f'\nExported {len(products_clean)} products to {products_output}')
print(products_clean.head(10))
print(products_clean.dtypes.rename('dtype').to_frame())

# --- PREPROCESS SALES DATA ---
sales_clean = sales_raw.drop_duplicates().copy()
sales_clean['date'] = pd.to_datetime(sales_clean['date'], errors='coerce', format='mixed')
invalid_date_rows = int(sales_clean['date'].isna().sum())
sales_clean = sales_clean.loc[sales_clean['date'].notna()].copy()
sales_clean['product'] = sales_clean['product'].astype('string').str.strip().replace(name_fixes)


def normalize_category(values, allowed):
    normalized = values.astype('string').str.strip().str.casefold()
    return normalized.map(allowed).fillna('Unknown')


sales_clean['client_type'] = normalize_category(sales_clean['client_type'], {
    'retail': 'Retail', 'wholesale': 'Wholesale', 'unknown': 'Unknown'
})
sales_clean['payment'] = normalize_category(sales_clean['payment'], {
    'cash': 'Cash', 'credit card': 'Credit card', 'transfer': 'Transfer', 'unknown': 'Unknown'
})

catalog = products_clean[['Product Name', 'Unit Price']]
sales_clean = sales_clean.merge(
    catalog, left_on='product', right_on='Product Name', how='left', validate='many_to_one'
)

if sales_clean['Unit Price'].isna().any():
    unmatched = sales_clean.loc[sales_clean['Unit Price'].isna(), 'product'].unique().tolist()
    raise ValueError(f'Sales products missing from catalog: {unmatched}')

sales_clean['unitprice'] = sales_clean['Unit Price'].astype('int64')
sales_clean['quantity'] = pd.to_numeric(sales_clean['quantity'], errors='raise').astype('int64')
sales_clean['total'] = pd.to_numeric(sales_clean['total'], errors='raise').astype('int64')

if not sales_clean['total'].eq(sales_clean['unitprice'] * sales_clean['quantity']).all():
    raise ValueError('A sales total does not match the corrected catalog price and quantity; review before export.')

sales_clean['date'] = sales_clean['date'].dt.strftime('%Y-%m-%d')
sales_clean = sales_clean.rename(columns={
    'product': 'product_name',
    'unitprice': 'unit_price',
    'payment': 'payment_method',
})

sales_clean = sales_clean[['date', 'client_type', 'product_name', 'unit_price', 'quantity', 'total', 'payment_method']]
sales_output = OUTPUT_DIR / 'MotorPH_Sales_Cleaned_Q3_2025.csv'
sales_clean.to_csv(sales_output, index=False)

print(f'\nRemoved {invalid_date_rows} rows with missing or invalid dates.')
print(f'Exported {len(sales_clean)} cleaned sales rows to {sales_output}')
print(sales_clean.head(10))
print(sales_clean.dtypes.rename('dtype').to_frame())

# --- VALIDATIONS ---
required_product_columns = [
    'Product ID Number', 'Product Name', 'Product Type', 'Unit Price',
    'Date of Manufacturing', 'Date of Acquisition'
]

assert products_clean.columns.tolist() == required_product_columns
assert products_clean['Product ID Number'].tolist() == list(range(1, len(products_clean) + 1))
assert products_clean[required_product_columns].notna().all().all()
assert products_clean['Date of Manufacturing'].between(1900, 2026).all()
assert products_clean['Date of Acquisition'].between(1900, 2026).all()
assert not products_clean.duplicated().any()
assert not sales_clean.duplicated().any()
assert sales_clean['product_name'].isin(products_clean['Product Name']).all()
assert sales_clean['total'].eq(sales_clean['unit_price'] * sales_clean['quantity']).all()

print('\nAll validations passed.')
print(f'Product report: {len(products_clean)} rows; columns are in the required order.')
print(f'Sales export: {len(sales_clean)} rows; dates are valid ISO-formatted dates.')
print(f'Product report CSV: {products_output}')
print(f'Sales CSV: {sales_output}')