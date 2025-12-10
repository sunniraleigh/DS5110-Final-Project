# Kaylee Faherty
# Overview Data Analysis - DS5110 Project

#Import Packages
#For Data Handlling & Transformation
import pandas as pd
import numpy as np
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
#Statistical Testing & Modeling
from statsmodels.stats.proportion import proportions_ztest
import scipy.stats as stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

# Increase default font sizes for all graphs
plt.rcParams.update({
    "font.size": 20,        # base font size
    "axes.titlesize": 24,   # title font size
    "axes.labelsize": 20,   # x and y axis labels
    "xtick.labelsize": 20,  # x tick labels
    "ytick.labelsize": 20,  # y tick labels
    "legend.fontsize": 20   # legend text
})

# Load dataset
df = pd.read_csv("combined_cleaned_subscriber_data.csv", low_memory=False)
df.columns = df.columns.str.strip().str.lower()

# Standardize city names to title case
df['city'] = df['city'].astype(str).str.strip().str.title()

# Ensure columns exist before filtering
if 'state' in df.columns:
    df = df[df['state'].astype(str).str.strip().str.upper() == "ME"].copy()

# Filter to Maine only - note that account ID column header in files given was spelled "accoutid"
df = df[df['state'].str.strip().str.upper() == "ME"].copy()
print(f"Dataset filtered to Maine only: {df['city'].nunique()} cities, {df['accoutid'].nunique()} subscribers")

# Build period column + churn flag
# Churn Flag - Identifies subscribers whose last seen period is before the dataset's max period
# Map month names to numbers
month_map = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
}
df['file_month'] = df['file_month'].astype(str).str.strip().str.lower().map(month_map)
df['file_year'] = pd.to_numeric(df['file_year'], errors='coerce')

# Only keep rows with valid month/year before building period
df = df[df['file_month'].notna() & df['file_year'].notna()].copy()

# Build period as integer to avoid float suffix (e.g. .0)
df['period'] = (df['file_year'].astype(int) * 100 + df['file_month'].astype(int)).astype(int)

# Parse period_dt safely
df['period_dt'] = pd.to_datetime(df['period'].astype(str), format='%Y%m', errors='coerce')

# --- Build churn flag ---
last_seen = df.groupby('accoutid')['period'].max().reset_index()
last_seen = last_seen.rename(columns={'period':'last_seen_period'})
max_period = df['period'].max()
last_seen['cancelled'] = (last_seen['last_seen_period'] < max_period).astype(int)
df = df.merge(last_seen[['accoutid','cancelled']], on='accoutid', how='left')

required_cols = ['publication_name', 'is_digital', 'bill_method', 'city', 'rate_code', 'accoutid', 'cancelled']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print("Warning: missing columns in dataset:", missing)

# Churn rate summary tables
# Overall Churn Rate
overall_total = df['accoutid'].nunique()
overall_cancelled = df[df['cancelled'] == 1]['accoutid'].nunique()
overall_churn_rate = overall_cancelled / overall_total * 100

print("\n=== Overall Churn Rate ===")
print(f"Total Subscribers: {overall_total}")
print(f"Cancelled Subscribers: {overall_cancelled}")
print(f"Overall Churn Rate: {overall_churn_rate:.1f}%")

# 1. By Publication
pub_total = df.groupby('publication_name')['accoutid'].nunique()
pub_cancelled = df[df['cancelled']==1].groupby('publication_name')['accoutid'].nunique()
pub_summary = pd.DataFrame({'Cancelled': pub_cancelled, 'Total': pub_total})
pub_summary['Cancelled'] = pub_summary['Cancelled'].fillna(0)
pub_summary['Churn Rate (%)'] = (pub_summary['Cancelled'] / pub_summary['Total'] * 100).round(1)
print("\n=== Churn Rate by Publication ===")
print(pub_summary.sort_values('Churn Rate (%)', ascending=False).to_string())

# 2. By Format
fmt_total = df.groupby('is_digital')['accoutid'].nunique()
fmt_cancelled = df[df['cancelled']==1].groupby('is_digital')['accoutid'].nunique()
fmt_summary = pd.DataFrame({'Cancelled': fmt_cancelled, 'Total': fmt_total})
fmt_summary['Cancelled'] = fmt_summary['Cancelled'].fillna(0)
fmt_summary['Churn Rate (%)'] = (fmt_summary['Cancelled'] / fmt_summary['Total'] * 100).round(1)
fmt_summary.index = fmt_summary.index.map({True:'Digital', False:'Print'})
print("\n=== Churn Rate by Format ===")
print(fmt_summary.to_string())

# 3. By Bill Method
bill_total = df.groupby('bill_method')['accoutid'].nunique()
bill_cancelled = df[df['cancelled']==1].groupby('bill_method')['accoutid'].nunique()
bill_summary = pd.DataFrame({'Cancelled': bill_cancelled, 'Total': bill_total})
bill_summary['Cancelled'] = bill_summary['Cancelled'].fillna(0)
bill_summary['Churn Rate (%)'] = (bill_summary['Cancelled'] / bill_summary['Total'] * 100).round(1)
print("\n=== Churn Rate by Bill Method ===")
print(bill_summary.sort_values('Churn Rate (%)', ascending=False).to_string())

# 4. By cities (≥10 subscribers) - Shows top 15 churn rates
city_total = df.groupby('city')['accoutid'].nunique()
city_cancelled = df[df['cancelled']==1].groupby('city')['accoutid'].nunique()
city_total = city_total[city_total >= 10]
city_summary = pd.DataFrame({'Cancelled': city_cancelled, 'Total': city_total})
city_summary['Cancelled'] = city_summary['Cancelled'].fillna(0)
city_summary['Churn Rate (%)'] = (city_summary['Cancelled'] / city_summary['Total'] * 100).round(1)
city_summary = city_summary.dropna().sort_values('Churn Rate (%)', ascending=False).head(15)
print("\n=== Top 15 Cities by Churn Rate (≥10 Subscribers) ===")
print(city_summary.to_string())

# 5. By Rate Code (≥10 subscribers) - Shows top 15 churn rates
rate_total = df.groupby('rate_code')['accoutid'].nunique()
rate_cancelled = df[df['cancelled']==1].groupby('rate_code')['accoutid'].nunique()
rate_total = rate_total[rate_total >= 10]
rate_summary = pd.DataFrame({'Cancelled': rate_cancelled, 'Total': rate_total})
rate_summary['Cancelled'] = rate_summary['Cancelled'].fillna(0)
rate_summary['Churn Rate (%)'] = (rate_summary['Cancelled'] / rate_summary['Total'] * 100).round(1)
rate_summary = rate_summary.dropna().sort_values('Churn Rate (%)', ascending=False).head(15)
print("\n=== Top 15 Rate Codes by Churn Rate (≥10 Subscribers) ===")
print(rate_summary.to_string())

#Confidence intervals for churn rates
#Helper function for churn CI
def churn_ci(cancelled, total, alpha=0.05):
    """
    Calculate churn rate and confidence interval (Wilson score).
    cancelled: number of cancelled subscribers
    total: total subscribers
    alpha: significance level (default 0.05 for 95% CI)
    """
    if total == 0:
        return (np.nan, np.nan, np.nan)
    rate = cancelled / total
    ci_low, ci_high = proportion_confint(count=cancelled, nobs=total, alpha=alpha, method='wilson')
    return (rate * 100, ci_low * 100, ci_high * 100)

#Overall churn rate with CI
overall_total = df['accoutid'].nunique()
overall_cancelled = df[df['cancelled'] == 1]['accoutid'].nunique()
overall_rate, overall_low, overall_high = churn_ci(overall_cancelled, overall_total)

print("\n=== Overall Churn Rate (with 95% CI) ===")
print(f"Total Subscribers: {overall_total}")
print(f"Cancelled Subscribers: {overall_cancelled}")
print(f"Overall Churn Rate: {overall_rate:.1f}% (95% CI: {overall_low:.1f}% – {overall_high:.1f}%)")

#Add CI columns to each summary table
for summary_name, summary_df in {
    "Publication": pub_summary,
    "Format": fmt_summary,
    "Bill Method": bill_summary,
    "City": city_summary,
    "Rate Code": rate_summary
}.items():
    summary_df[['CI Low (%)','CI High (%)']] = summary_df.apply(
    lambda row: pd.Series(churn_ci(row['Cancelled'], row['Total']))[1:],  # take only low/high
    axis=1
)
    print(f"\n=== {summary_name} Churn Rates with 95% CI ===")
    print(summary_df.to_string())

#Bar charts with churn% and subscriber counts
def plot_churn_with_ci(summary_df, category_name):
    """
    Creates a bar chart with subscriber counts, churn %, and confidence intervals.
    Uses pre-calculated CI columns: 'CI Low (%)' and 'CI High (%)'.
    summary_df: DataFrame with columns ['Cancelled','Total','Churn Rate (%)','CI Low (%)','CI High (%)']
    category_name: str, name of the category (e.g., 'Publication', 'Format')
    """

    fig, ax1 = plt.subplots(figsize=(10,6))

    # Bar chart for churn rate
    sns.barplot(
        x=summary_df.index,
        y="Churn Rate (%)",
        data=summary_df,
        ax=ax1,
        color="skyblue",
        edgecolor="black"
    )

    # Add error bars using pre-calculated CI bounds
    ax1.errorbar(
        x=range(len(summary_df)),
        y=summary_df["Churn Rate (%)"],
        yerr=[
            (summary_df["Churn Rate (%)"] - summary_df["CI Low (%)"]).abs(),
            (summary_df["CI High (%)"] - summary_df["Churn Rate (%)"]).abs()
        ],
        fmt="none",
        c="black",
        capsize=5
    )

    # Secondary axis for subscriber counts
    ax2 = ax1.twinx()
    ax2.plot(
        range(len(summary_df)),
        summary_df["Total"],
        color="red",
        marker="o",
        linestyle="--",
        label="Subscriber Count"
    )
    ax2.set_ylabel("Subscriber Count", color="red")

    # Labels and formatting
    ax1.set_title(f"Churn Rate with 95% CI by {category_name}")
    ax1.set_ylabel("Churn Rate (%)")
    ax1.set_xlabel(category_name)

    # Conditional formatting for x-axis labels
    if category_name in ["Publication", "Bill Method"]:
        # Wrap and angle labels
        labels = [label.get_text().replace(" ", "\n") for label in ax1.get_xticklabels()]
        ax1.set_xticks(ax1.get_xticks())
        ax1.set_xticklabels(labels, rotation=55)
    else:
        ax1.tick_params(axis='x', rotation=0)

    # Legend placement depends on category
    if category_name == "Bill Method":
        ax2.legend(loc="upper right")
    else:
        ax2.legend(loc="upper left")


    plt.tight_layout()
    plt.show()

plot_churn_with_ci(pub_summary, "Publication")
plot_churn_with_ci(fmt_summary, "Format")
plot_churn_with_ci(bill_summary, "Bill Method")

# P-value calculations
pval_results = {}

# 1. Publication (Chi-square test)
pub_contingency = pd.crosstab(df['publication_name'], df['cancelled'])
chi2, pval, dof, expected = stats.chi2_contingency(pub_contingency)
pval_results['Publication'] = pval

# 2. Format (Digital vs Print, two-proportion z-test)
stat, pval = proportions_ztest(count=fmt_summary['Cancelled'].values, nobs=fmt_summary['Total'].values)
pval_results['Format (Digital vs Print)'] = pval

# 3. Bill Method (Chi-square test)
bill_contingency = pd.crosstab(df['bill_method'], df['cancelled'])
chi2, pval, dof, expected = stats.chi2_contingency(bill_contingency)
pval_results['Bill Method'] = pval

# 4. Cities (Chi-square test, ≥10 subs)
city_contingency = pd.crosstab(df[df['city'].isin(city_total.index)]['city'], df['cancelled'])
chi2, pval, dof, expected = stats.chi2_contingency(city_contingency)
pval_results['Cities (≥10 subs)'] = pval

# 5. Rate Codes (Chi-square test, ≥10 subs)
rate_contingency = pd.crosstab(df[df['rate_code'].isin(rate_total.index)]['rate_code'], df['cancelled'])
chi2, pval, dof, expected = stats.chi2_contingency(rate_contingency)
pval_results['Rate Codes (≥10 subs)'] = pval

# Build summary table
pval_table = pd.DataFrame.from_dict(pval_results, orient='index', columns=['p-value'])
pval_table['Significant (<0.05)'] = pval_table['p-value'] < 0.05
print("\n=== P-Values for Churn Rate Differences ===")
print(pval_table.to_string())

# Helper: Cramér's V for chi-square effect size
def cramers_v(chi2, n, r, c):
    """
    Calculate Cramér's V effect size.
    chi2: chi-square statistic
    n: total sample size
    r: number of rows in contingency table
    c: number of columns in contingency table
    """
    return np.sqrt(chi2 / (n * (min(r-1, c-1))))

# Extended results dictionary
# Deduplicate so each subscriber is counted once
unique_df = df.drop_duplicates('accoutid')

extended_results = {}

# 1. Publication (Chi-square)
pub_contingency = pd.crosstab(unique_df['publication_name'], unique_df['cancelled'])
chi2, pval, dof, expected = stats.chi2_contingency(pub_contingency)
n = pub_contingency.sum().sum()
cramerv = cramers_v(chi2, n, *pub_contingency.shape)
extended_results['Publication'] = {'chi2': chi2, 'pval': pval, 'cramers_v': cramerv}

# 2. Format (Z-test)
fmt_counts = unique_df.groupby('is_digital')['cancelled'].sum()
fmt_totals = unique_df.groupby('is_digital')['accoutid'].nunique()
stat, pval = proportions_ztest(count=fmt_counts.values, nobs=fmt_totals.values)
extended_results['Format (Digital vs Print)'] = {'z': stat, 'pval': pval}

# 3. Bill Method (Chi-square)
bill_contingency = pd.crosstab(unique_df['bill_method'], unique_df['cancelled'])
chi2, pval, dof, expected = stats.chi2_contingency(bill_contingency)
n = bill_contingency.sum().sum()
cramerv = cramers_v(chi2, n, *bill_contingency.shape)
extended_results['Bill Method'] = {'chi2': chi2, 'pval': pval, 'cramers_v': cramerv}

# 4. Cities (Chi-square, ≥10 subs)
city_subset = unique_df[unique_df['city'].isin(city_total.index)]
city_contingency = pd.crosstab(city_subset['city'], city_subset['cancelled'])
chi2, pval, dof, expected = stats.chi2_contingency(city_contingency)
n = city_contingency.sum().sum()
cramerv = cramers_v(chi2, n, *city_contingency.shape)
extended_results['Cities (≥10 subs)'] = {'chi2': chi2, 'pval': pval, 'cramers_v': cramerv}

# 5. Rate Codes (Chi-square, ≥10 subs)
rate_subset = unique_df[unique_df['rate_code'].isin(rate_total.index)]
rate_contingency = pd.crosstab(rate_subset['rate_code'], rate_subset['cancelled'])
chi2, pval, dof, expected = stats.chi2_contingency(rate_contingency)
n = rate_contingency.sum().sum()
cramerv = cramers_v(chi2, n, *rate_contingency.shape)
extended_results['Rate Codes (≥10 subs)'] = {'chi2': chi2, 'pval': pval, 'cramers_v': cramerv}

# Build extended summary table
ext_table = pd.DataFrame(extended_results).T

# Replace NaN values with "N/A" for clarity
ext_table = ext_table.fillna("N/A")

# Multiple testing correction (Bonferroni and FDR)
pvals = ext_table['pval'].dropna().values  # drop NaN for tests without pval
bonferroni = multipletests(pvals, alpha=0.05, method='bonferroni')[1]
fdr = multipletests(pvals, alpha=0.05, method='fdr_bh')[1]

ext_table.loc[ext_table['pval'].notna(), 'Bonferroni adj p'] = bonferroni
ext_table.loc[ext_table['pval'].notna(), 'FDR adj p'] = fdr

print("\n=== Extended Statistical Results ===")
print(ext_table.to_string())
print("\nNote: Cramér's V reported for chi-square tests as effect size. "
      "Chi² and z statistics shown alongside p-values. "
      "Bonferroni and FDR corrections applied for multiple testing.")

print("\n=== Interpretation Guide for Extended Results ===")
print("How to read this table:")
print("- p-value: Probability that observed churn differences are due to chance.")
print("    • <0.05 usually considered statistically significant.")
print("- Chi² / z statistic: Test statistic showing strength of evidence against the null hypothesis.")
print("    • Higher values = stronger evidence of real differences.")
print("- Cramér's V (effect size for chi-square tests):")
print("    • ~0.1 = small association, ~0.3 = medium, ~0.5+ = large.")
print("- Bonferroni adjusted p-value: Conservative correction for multiple tests.")
print("    • Helps avoid false positives, but may miss weaker real effects.")
print("- FDR (Benjamini–Hochberg): Balances false discovery risk.")
print("    • More power to detect true effects while controlling overall error rate.")



# Additional Visuals

# 1. Top 5 Cities by Publication (October 2024)
october_df = df[df['period'] == 202410]
city_counts = (
    october_df.groupby(['publication_name','city'])['accoutid']
    .nunique()
    .reset_index(name='subscriber_count')
)
pub_totals = (
    october_df.groupby('publication_name')['accoutid']
    .nunique()
    .reset_index(name='total_subscribers')
)
city_counts = city_counts.merge(pub_totals, on='publication_name')
city_counts['percent'] = city_counts['subscriber_count'] / city_counts['total_subscribers'] * 100
city_counts = city_counts.sort_values(['publication_name','subscriber_count'], ascending=[True,False])
top5_cities = city_counts.groupby('publication_name').head(5)

pivot = top5_cities.pivot_table(index='publication_name', columns='city', values='percent', fill_value=0)
ax = pivot.plot(kind='bar', stacked=True, figsize=(12,6), colormap='tab20')
plt.title("Top 5 Cities by Publication (October 2024)")
plt.ylabel("Percentage of Subscribers")
plt.xlabel("Publication")
labels = [label.get_text().replace(" ", "\n") for label in ax.get_xticklabels()]
ax.set_xticklabels(labels)
for i, pub in enumerate(pivot.index):
    cumulative = 0
    for city in pivot.columns:
        value = pivot.loc[pub, city]
        if value > 0:
            # Position annotation in the middle of the segment
            ax.text(i, cumulative + value/2, city,
                    ha='center', va='center', fontsize=18, color='black')
            cumulative += value
ax.get_legend().remove()
plt.tight_layout()
plt.show()

# 2. Digital vs Print Subscribers by Publication (October 2024)
df_last = df[df['period'] == df['period'].max()].copy()
df_last['format'] = df_last['is_digital'].map({True: 'Digital', False: 'Print'})
unique_counts = (
    df_last.groupby(['publication_name','format'])['accoutid']
    .nunique()
    .reset_index(name='count')
)

plt.figure(figsize=(12,6))
ax = sns.barplot(x="publication_name", y="count", hue="format", data=unique_counts, palette="Set2")
plt.title("Digital vs Print Subscribers by Publication (Latest Month)")
plt.ylabel("Unique Subscribers")
plt.xlabel("Publication")

# Wrap labels and rotate them 90 degrees
labels = [label.get_text().replace(" ", "\n") for label in ax.get_xticklabels()]
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(labels, rotation=90)

plt.legend(title="Format")
plt.tight_layout()
plt.show()

# 3. Subscriber Counts Over Time
df['period_dt'] = pd.to_datetime(df['period'].astype(str), format='%Y%m')
subs_by_period = df.groupby('period_dt')['accoutid'].nunique()
plt.figure(figsize=(15,8))
ax = subs_by_period.plot(kind="line", marker="o")
plt.title("Subscriber Counts Over Time")
plt.xlabel("Month")
plt.ylabel("Unique Subscribers")
ax.set_xticks(subs_by_period.index)
ax.set_xticklabels(subs_by_period.index.strftime('%b %Y'), rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 4. Churn Rate Over Time
churn_by_period = df.groupby('period_dt')['cancelled'].mean() * 100
plt.figure(figsize=(15,8))
ax = churn_by_period.plot(kind="line", marker="o", color="red")
plt.title("Churn Rate Over Time (%)")
plt.xlabel("Month")
plt.ylabel("Churn Rate (%)")
ax.set_xticks(churn_by_period.index)
ax.set_xticklabels(churn_by_period.index.strftime('%b %Y'), rotation=45, ha="right")
plt.tight_layout()
plt.show()

