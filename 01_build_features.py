"""
Step 1, Data engineering with SQL.
Loads the Online Retail II transactions into a SQLite database, then uses SQL to
build a customer-level feature table:
  * Features come from a CALIBRATION window (everything up to a cutoff date).
  * The TARGET is each customer's revenue in the next 91 days (PREDICTION window).
This is the standard predictive-LTV setup: learn from past behaviour, predict
future value.

Run:  python 01_build_features.py
Outputs: data/retail.db, data/customer_features.csv, sql/build_features.sql
"""
import sqlite3
from pathlib import Path
import pandas as pd

DATA = Path("data"); SQLDIR = Path("sql")
PRED_WINDOW_DAYS = 91          # predict the next ~quarter
RECENT_WINDOW_DAYS = 90        # "recent activity" window before the cutoff

con = sqlite3.connect(DATA / "retail.db")
has_table = con.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'"
).fetchone()

# ---------------------------------------------------------------------------
# 1. Load + clean the raw transactions (only if not already in the database)
# ---------------------------------------------------------------------------
if not has_table:
    print("Reading Excel (this takes a minute) ...")
    sheets = pd.read_excel(DATA / "online_retail_II.xlsx", sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Customer ID": "customer_id", "Price": "price",
                            "Invoice": "invoice", "StockCode": "stock_code",
                            "Quantity": "quantity", "InvoiceDate": "invoice_date",
                            "Country": "country"})
    before = len(df)
    df = df.dropna(subset=["customer_id"])                   # need a customer
    df = df[~df["invoice"].astype(str).str.startswith("C")]  # drop cancellations
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]        # drop returns/bad rows
    df["revenue"] = df["quantity"] * df["price"]
    df["customer_id"] = df["customer_id"].astype(int)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    print(f"Rows: {before:,} raw -> {len(df):,} clean | customers: {df.customer_id.nunique():,}")
    df[["customer_id", "invoice", "stock_code", "quantity", "price", "revenue",
        "invoice_date", "country"]].assign(
        invoice_date=lambda d: d["invoice_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ).to_sql("transactions", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_cust ON transactions(customer_id)")
else:
    print("Reusing existing data/retail.db")

# Cutoff = last date in the data minus the prediction window
max_date = pd.to_datetime(con.execute("SELECT MAX(invoice_date) FROM transactions").fetchone()[0])
cutoff = (max_date - pd.Timedelta(days=PRED_WINDOW_DAYS)).strftime("%Y-%m-%d")
recent_start = (pd.to_datetime(cutoff) - pd.Timedelta(days=RECENT_WINDOW_DAYS)).strftime("%Y-%m-%d")
print(f"Data ends {max_date.date()} | cutoff = {cutoff} "
      f"(calibration < cutoff, target = next {PRED_WINDOW_DAYS} days)")

# ---------------------------------------------------------------------------
# 3. Feature engineering in SQL
# ---------------------------------------------------------------------------
sql = f"""
WITH cal AS (   -- calibration window: behaviour we learn from
    SELECT * FROM transactions WHERE invoice_date < '{cutoff}'
),
pred AS (       -- prediction window: the value we want to forecast
    SELECT customer_id, SUM(revenue) AS future_revenue
    FROM transactions
    WHERE invoice_date >= '{cutoff}'
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    CAST(julianday('{cutoff}') - julianday(MAX(c.invoice_date)) AS INT) AS recency_days,
    CAST(julianday('{cutoff}') - julianday(MIN(c.invoice_date)) AS INT) AS tenure_days,
    COUNT(DISTINCT c.invoice)                         AS frequency,
    ROUND(SUM(c.revenue), 2)                          AS monetary_total,
    ROUND(SUM(c.revenue) * 1.0 / COUNT(DISTINCT c.invoice), 2) AS avg_order_value,
    SUM(c.quantity)                                   AS total_items,
    COUNT(DISTINCT c.stock_code)                      AS distinct_products,
    COUNT(DISTINCT strftime('%Y-%m', c.invoice_date)) AS active_months,
    ROUND(SUM(CASE WHEN c.invoice_date >= '{recent_start}' THEN c.revenue ELSE 0 END), 2)
                                                      AS recent_revenue_90d,
    COALESCE(ROUND(p.future_revenue, 2), 0)           AS future_revenue
FROM cal c
LEFT JOIN pred p ON c.customer_id = p.customer_id
GROUP BY c.customer_id;
"""
SQLDIR.mkdir(exist_ok=True)
(SQLDIR / "build_features.sql").write_text(sql.strip() + "\n")

feats = pd.read_sql(sql, con)
con.close()

feats.to_csv(DATA / "customer_features.csv", index=False)
print(f"\nCustomer feature table: {feats.shape[0]:,} customers x {feats.shape[1]} cols")
print(f"Customers active in prediction window: "
      f"{(feats.future_revenue > 0).mean():.1%}")
print("Saved data/customer_features.csv and sql/build_features.sql")
