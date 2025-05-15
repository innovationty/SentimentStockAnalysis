# step4_analysis.py
import pandas as pd, statsmodels.api as sm
import matplotlib.pyplot as plt, seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv("merged.csv", parse_dates=["date"])
TICKERS = sorted(df["Stock Name"].unique())

rows = []
for tic in TICKERS:
    sub = df[df["Stock Name"] == tic]
    if sub.empty:
        continue

    # Pearson correlation
    r = sub["sent_roll5"].corr(sub["next_ret"])

    # OLS
    X = sm.add_constant(sub["sent_roll5"])
    y = sub["next_ret"]
    ols = sm.OLS(y, X).fit()
    p = ols.pvalues["sent_roll5"]

    rows.append({"ticker": tic, "pearson": r, "p_value": p})
    print(f"\n[{tic}] Pearson r={r:.4f}, p={p:.3f}")
    print(ols.summary().tables[1])

pd.DataFrame(rows).to_csv("summary_stats.csv", index=False)
print("\n✅ Generated summary_stats.csv")
# Load previously saved results
results_df = pd.DataFrame(rows)

# Sort by absolute Pearson correlation
results_df['abs_pearson'] = results_df['pearson'].abs()
results_df_sorted = results_df.sort_values('abs_pearson', ascending=False)

# Create significance flag
results_df_sorted['significant'] = results_df_sorted['p_value'] < 0.05
results_df_sorted['sig_label'] = results_df_sorted['significant'].apply(lambda x: '*' if x else '')

# Set chart style and size
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Color mapping: significant = red, not significant = blue
colors = ['#E74C3C' if sig else '#3498DB' for sig in results_df_sorted['significant']]

# Draw bar chart
bars = plt.bar(results_df_sorted['ticker'], results_df_sorted['pearson'], color=colors)

# Add horizontal zero line
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Add significance asterisks
for i, (idx, row) in enumerate(results_df_sorted.iterrows()):
    if row['significant']:
        plt.text(
            i,
            row['pearson'] + (0.01 if row['pearson'] >= 0 else -0.01),
            '*',
            ha='center',
            fontsize=14,
            color='red'
        )

# Add title and labels
plt.title('Pearson Correlation between Social Media Sentiment and Next-Day Stock Returns', fontsize=16)
plt.xlabel('Ticker', fontsize=12)
plt.ylabel('Pearson Correlation Coefficient', fontsize=12)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E74C3C', label='Significant (p<0.05)'),
    Patch(facecolor='#3498DB', label='Not Significant (p≥0.05)')
]
plt.legend(handles=legend_elements, loc='best')

# Rotate x-axis labels
plt.xticks(rotation=45)

# Annotate bars with exact values
for i, (idx, row) in enumerate(results_df_sorted.iterrows()):
    plt.text(
        i,
        row['pearson'] + (0.01 if row['pearson'] >= 0 else -0.01),
        f"{row['pearson']:.3f}",
        ha='center',
        va='bottom' if row['pearson'] >= 0 else 'top',
        fontsize=8
    )

plt.tight_layout()
plt.savefig("pearson_correlation_analysis.png", dpi=300)
print("✅ Generated Pearson correlation chart → pearson_correlation_analysis.png")

# Generate detailed results table
results_table = results_df_sorted.copy()
results_table['pearson'] = results_table['pearson'].round(4)
results_table['p_value'] = results_table['p_value'].round(4)
results_table = results_table[['ticker', 'pearson', 'p_value', 'significant']]
results_table.columns = ['Ticker', 'Pearson', 'P-value', 'Significant (p<0.05)']

# Save to CSV
results_table.to_csv("pearson_correlation_table.csv", index=False, encoding='utf-8-sig')
print("✅ Generated Pearson correlation table → pearson_correlation_table.csv")

# Print statistical summary
sig_count = results_df_sorted['significant'].sum()
pos_count = (results_df_sorted['pearson'] > 0).sum()
neg_count = (results_df_sorted['pearson'] < 0).sum()

print("\n===== Pearson Correlation Analysis Summary =====")
print(f"Total tickers: {len(results_df_sorted)}")
print(f"Significant correlations: {sig_count} ({sig_count/len(results_df_sorted)*100:.1f}%)")
print(f"Positive correlations: {pos_count} ({pos_count/len(results_df_sorted)*100:.1f}%)")
print(f"Negative correlations: {neg_count} ({neg_count/len(results_df_sorted)*100:.1f}%)")
print(f"Maximum positive correlation: {results_df_sorted['pearson'].max():.4f}")
print(f"Maximum negative correlation: {results_df_sorted['pearson'].min():.4f}")
print(f"Average correlation coefficient: {results_df_sorted['pearson'].mean():.4f}")
print(f"Standard deviation of correlations: {results_df_sorted['pearson'].std():.4f}")


# ==== Scatter Plot Analysis for Selected Stocks ====
print("\n===== Creating Scatter Plot Analysis =====")

# Add significant flag to results_df or use results_df_sorted
results_df['significant'] = results_df['p_value'] < 0.05

# Select representative stocks: highest correlations and significant ones
top_abs_corr = results_df.nlargest(2, 'abs_pearson')['ticker'].tolist()  # Top 2 absolute correlations
sig_stocks = results_df[results_df['significant']]['ticker'].tolist()  # Significant correlations

# Ensure no duplicates
selected_stocks = list(set(top_abs_corr + sig_stocks))
print(f"Selected stocks for scatter plot: {', '.join(selected_stocks)}")

if selected_stocks:
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Prepare data
    scatter_data = pd.DataFrame()
    for tic in selected_stocks:
        sub_data = df[df["Stock Name"] == tic][["sent_roll5", "next_ret"]].copy()
        sub_data['Stock'] = tic
        scatter_data = pd.concat([scatter_data, sub_data])
    
    # Create scatter plot
    scatter_plot = sns.scatterplot(
        data=scatter_data, 
        x="sent_roll5", 
        y="next_ret", 
        hue="Stock",
        style="Stock",
        s=50,
        alpha=0.7
    )
    
    # Add regression lines for each stock
    for tic in selected_stocks:
        sub_data = scatter_data[scatter_data['Stock'] == tic]
        r = results_df[results_df['ticker'] == tic]['pearson'].values[0]
        p = results_df[results_df['ticker'] == tic]['p_value'].values[0]
        sig_mark = '*' if p < 0.05 else ''
        
        sns.regplot(
            data=sub_data,
            x="sent_roll5", 
            y="next_ret", 
            scatter=False,
            label=f"{tic} (r={r:.3f}{sig_mark}, p={p:.3f})"
        )
    
    # Add overall trend line
    sns.regplot(
        data=scatter_data, 
        x="sent_roll5", 
        y="next_ret", 
        scatter=False, 
        color='black',
        line_kws={"linestyle": "--", "linewidth": 1.5},
        label="Overall trend"
    )
    
    # Add chart elements
    plt.title("Social Media Sentiment vs. Next-Day Stock Returns", fontsize=16)
    plt.xlabel("5-day Rolling Sentiment Index", fontsize=12)
    plt.ylabel("Next-Day Return (%)", fontsize=12)
    plt.grid(alpha=0.3)
    
    # Add zero lines
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Adjust legend to show correlation values
    plt.legend(title="Stock (correlation)", loc='best')
    
    # Save figure
    plt.tight_layout()
    plt.savefig("sentiment_returns_scatter.png", dpi=300)
    print("✅ Created sentiment vs. returns scatter plot → sentiment_returns_scatter.png")