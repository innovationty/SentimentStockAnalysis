# step2_rolling.py
import pandas as pd

ROLL_N = 5                         # Rolling window size, adjustable

df = pd.read_csv("daily_sentiment.csv", parse_dates=["date"])

# Ensure sorted by stock & date
df = df.sort_values(["Stock Name", "date"])

# Calculate rolling mean separately for each stock; use available mean if less than 5 days
df["sent_roll5"] = (
    df.groupby("Stock Name")["sent_index"]
      .transform(lambda s: s.rolling(ROLL_N, min_periods=1).mean())
)

df.to_csv("daily_sentiment.csv", index=False)   # Overwrite or save with new name
print("✅ Added 5-day rolling sentiment mean for each stock → daily_sentiment.csv")
print(df.head(8))