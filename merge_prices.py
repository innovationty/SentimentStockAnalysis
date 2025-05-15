# step3_merge_prices.py  —— Stooq data source
import pandas as pd
from pandas_datareader.data import DataReader
from datetime import timedelta

# ===== 1. Load sentiment data =====
sent = pd.read_csv("daily_sentiment.csv", parse_dates=["date"])
TICKERS = sorted(sent["Stock Name"].unique())          # Auto detect stocks

start = sent["date"].min()
end   = sent["date"].max() + timedelta(days=1)

# ===== 2. Download closing prices (Stooq) =====
frames = []
bad    = []
for t in TICKERS:
    try:
        close = DataReader(t, "stooq", start, end)["Close"]   # Series
        close.name = t
        frames.append(close)
    except Exception as e:
        print(f"⚠ No data from Stooq: {t} ({e})")
        bad.append(t)

if not frames:
    raise RuntimeError("All tickers have no data in Stooq, cannot proceed.")

price_close = pd.concat(frames, axis=1).sort_index()   # columns = different stocks

# ===== 3. Calculate next-day returns =====
ret = (
    price_close.pct_change()
               .shift(-1)              # next day returns
               .dropna()
               .stack()
               .to_frame("next_ret")
               .reset_index()
               .rename(columns={"level_1": "Stock Name", "Date": "date"})
)

ret["date"] = ret["date"].dt.date
sent["date"] = pd.to_datetime(sent["date"]).dt.date
# ===== 4. Merge data =====
merged = sent.merge(ret, on=["date", "Stock Name"], how="inner")
merged.to_csv("merged.csv", index=False)

print(f"✅ Merge completed → merged.csv  Rows: {len(merged)}  Stocks: {len(TICKERS) - len(bad)}")
if bad:
    print("Following stocks not found in Stooq, skipped:", ", ".join(bad))
print(merged.head())