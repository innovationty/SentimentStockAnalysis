# step1_daily_sentiment.py
import pandas as pd

df = pd.read_csv("tweets_with_sentiment.csv", parse_dates=["Date"])
df["date"] = df["Date"].dt.date          # Keep date only

# Group by date × stock × sentiment count
pivot = (
    df.groupby(["date", "Stock Name", "sentiment"])
      .size().unstack(fill_value=0)      # positive / negative / neutral columns
      .reset_index()
)

pivot["sent_index"] = (
    (pivot.get("positive", 0) - pivot.get("negative", 0)) / 
    (pivot.get("positive", 0) + pivot.get("negative", 0) + 0.1 * pivot.get("neutral", 0))
)

pivot.to_csv("daily_sentiment.csv", index=False)
print("✅ Daily sentiment index saved → daily_sentiment.csv")
print(pivot.head())