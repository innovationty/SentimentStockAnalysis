import pandas as pd
import statsmodels.api as sm

# 1. Read and sort data
df = pd.read_csv("merged.csv", parse_dates=["date"])
df = df.sort_values(["Stock Name", "date"])

# 2. Generate 1~5 day lagged returns
for k in range(1, 6):
    df[f"ret_lag{k}"] = df.groupby("Stock Name")["next_ret"].shift(k)

# 3. Remove rows with NaN values
df = df.dropna(subset=[f"ret_lag{k}" for k in range(1, 6)] + ["sent_index"])

# 4. Calculate Pearson correlation and run OLS
rows = []
for tic in df["Stock Name"].unique():
    sub = df[df["Stock Name"] == tic]
    # Correlations
    r_next = sub["sent_index"].corr(sub["next_ret"])
    print(f"[{tic}] Sentiment vs. Next-day Returns Pearson r={r_next:.4f}")
    # Regression
    X = sm.add_constant(sub["sent_index"])
    y = sub["next_ret"]
    model = sm.OLS(y, X).fit()
    print(model.summary().tables[1])
    rows.append({
        "ticker": tic,
        "pearson_next": r_next,
        "p_next": model.pvalues["sent_index"],
        "beta_sent": model.params["sent_index"],
        "R2": model.rsquared
    })

# 5. Save summary
pd.DataFrame(rows).to_csv("summary_stats.csv", index=False, encoding="utf-8-sig")
print("✅ Generated summary_stats.csv")

# Pearson Correlation Analysis Summary
print("\n======= Pearson Correlation Analysis =======")

results = pd.DataFrame(rows)

# Calculate correlations with lagged returns for each stock
corr_rows = []
for tic in df["Stock Name"].unique():
    sub = df[df["Stock Name"] == tic]
    
    # Calculate correlations for each lag period
    corrs = {}
    for k in range(1, 6):
        corr = sub["sent_index"].corr(sub[f"ret_lag{k}"])
        corrs[f"corr_lag{k}"] = corr
    
    # Calculate correlation with next-day returns
    corr_next = sub["sent_index"].corr(sub["next_ret"])
    
    print(f"\n===== {tic} - Pearson Correlation Coefficients =====")
    for k in range(1, 6):
        print(f"Sentiment correlation with {k}-day prior returns: {corrs[f'corr_lag{k}']:.4f}")
    print(f"Sentiment correlation with next-day returns: {corr_next:.4f}")
    
    # Store results
    corr_rows.append({
        "ticker": tic,
        "corr_lag1": corrs["corr_lag1"],
        "corr_lag2": corrs["corr_lag2"],
        "corr_lag3": corrs["corr_lag3"],
        "corr_lag4": corrs["corr_lag4"],
        "corr_lag5": corrs["corr_lag5"],
        "corr_next_day": corr_next
    })

# Create correlation dataframe
corr_df = pd.DataFrame(corr_rows)

# Calculate statistics for 1-day lag returns and sentiment correlation
lag1_corrs = corr_df["corr_lag1"]
sig_count = sum((results["p_next"] < 0.05))
pos_count = sum(lag1_corrs > 0)
neg_count = sum(lag1_corrs < 0)

print("\n===== Correlation Analysis Summary for Prior-Day Returns Impact on Sentiment =====")
print(f"Total stocks: {len(lag1_corrs)}")
print(f"Significant correlations: {sig_count} ({sig_count/len(lag1_corrs)*100:.1f}%)")
print(f"Positive correlations: {pos_count} ({pos_count/len(lag1_corrs)*100:.1f}%)")
print(f"Negative correlations: {neg_count} ({neg_count/len(lag1_corrs)*100:.1f}%)")
print(f"Maximum positive correlation: {lag1_corrs.max():.4f}")
print(f"Maximum negative correlation: {lag1_corrs.min():.4f}")
print(f"Average correlation coefficient: {lag1_corrs.mean():.4f}")
print(f"Standard deviation of correlations: {lag1_corrs.std():.4f}")

# Calculate average correlation for each lag period
lag_means = [corr_df[f'corr_lag{i}'].mean() for i in range(1, 6)]
print("\n===== Average Correlation Coefficients by Lag Period =====")
for i, mean in enumerate(lag_means):
    print(f"{i+1}-day lag: {mean:.4f}")

# Save correlation analysis results
corr_df.to_csv("lag_correlations.csv", index=False)
print("\n✅ Generated lag correlation analysis results → lag_correlations.csv")