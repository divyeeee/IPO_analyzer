#!/usr/bin/env python3
"""
Comprehensive IPO Data Analysis
Analyzes NSE IPO data from 2016-2025 covering listing gains, subscriptions, and trends.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'analysis_output'
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor': '#1a1a2e',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#b0b0b0',
    'ytick.color': '#b0b0b0',
    'axes.edgecolor': '#333355',
    'grid.color': '#2a2a4a',
    'grid.alpha': 0.5,
})

PALETTE = ['#00d4ff', '#ff6b6b', '#ffd93d', '#6bcb77', '#c084fc',
           '#ff9a3c', '#4ecdc4', '#f38181', '#a8e6cf', '#dcedc1']

# ── Load & Clean ────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(BASE_DIR / 'nse_ipo_merged.csv')
print(f"Total records: {len(df)}")

# Parse dates
for col in ['IPO Start Date', 'IPO End Date', 'Listing Date']:
    df[col] = pd.to_datetime(df[col], format='%d-%b-%Y', errors='coerce')

# Filter to only equity/SME IPOs that have listing data
equity_types = ['EQ', 'BE', 'SM', 'SME']
df_equity = df[df['Security Type'].isin(equity_types)].copy()
df_listed = df_equity[df_equity['Listing Day Gain (%)'].notna()].copy()

print(f"Equity/SME IPOs: {len(df_equity)}")
print(f"IPOs with listing gain data: {len(df_listed)}")

# ── Helper ──────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path.name}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("1. OVERVIEW STATISTICS")
print("="*70)

stats = {}

# Year-wise counts
year_counts = df_equity.groupby('IPO Year').size()
print(f"\nIPOs per year:\n{year_counts.to_string()}")

# Listing gain stats
lg = df_listed['Listing Day Gain (%)']
stats['total_ipos_with_listing_data'] = len(df_listed)
stats['mean_listing_gain'] = round(lg.mean(), 2)
stats['median_listing_gain'] = round(lg.median(), 2)
stats['positive_listing_pct'] = round((lg > 0).mean() * 100, 2)
stats['negative_listing_pct'] = round((lg < 0).mean() * 100, 2)
stats['max_listing_gain'] = round(lg.max(), 2)
stats['min_listing_gain'] = round(lg.min(), 2)

# Which IPO had the best/worst listing?
best_idx = lg.idxmax()
worst_idx = lg.idxmin()
stats['best_ipo'] = f"{df_listed.loc[best_idx, 'Company Name']} ({df_listed.loc[best_idx, 'IPO Year']}) → +{lg.max():.1f}%"
stats['worst_ipo'] = f"{df_listed.loc[worst_idx, 'Company Name']} ({df_listed.loc[worst_idx, 'IPO Year']}) → {lg.min():.1f}%"

for k, v in stats.items():
    print(f"  {k}: {v}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. IPO VOLUME BY YEAR (Bar Chart)
# ═══════════════════════════════════════════════════════════════════════════
print("\n2. IPO Volume by Year...")

fig, ax = plt.subplots(figsize=(12, 5))
years = sorted(df_equity['IPO Year'].dropna().unique())
counts = [len(df_equity[df_equity['IPO Year'] == y]) for y in years]

bars = ax.bar(years, counts, color=PALETTE[0], edgecolor='#0a0a1a', linewidth=0.5, width=0.7, alpha=0.9)
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(c),
            ha='center', va='bottom', fontweight='bold', color='#00d4ff', fontsize=10)

ax.set_xlabel('Year')
ax.set_ylabel('Number of IPOs')
ax.set_title('IPO Volume by Year (Equity & SME)')
ax.set_xticks(years)
ax.grid(axis='y', linestyle='--')
save_fig(fig, '01_ipo_volume_by_year')

# ═══════════════════════════════════════════════════════════════════════════
# 3. LISTING DAY GAIN DISTRIBUTION (Histogram + KDE)
# ═══════════════════════════════════════════════════════════════════════════
print("3. Listing Day Gain Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full distribution
ax = axes[0]
gains = df_listed['Listing Day Gain (%)'].clip(-100, 300)
ax.hist(gains, bins=60, color=PALETTE[0], alpha=0.7, edgecolor='#0a0a1a', linewidth=0.3)
ax.axvline(0, color='#ff6b6b', linestyle='--', linewidth=1.5, label='Break-even')
ax.axvline(gains.median(), color='#ffd93d', linestyle='--', linewidth=1.5, label=f'Median: {gains.median():.1f}%')
ax.set_xlabel('Listing Day Gain (%)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Listing Day Gains')
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--')

# Zoomed in -50% to 150%
ax = axes[1]
gains_zoom = df_listed[(df_listed['Listing Day Gain (%)'] >= -50) & (df_listed['Listing Day Gain (%)'] <= 150)]['Listing Day Gain (%)']
ax.hist(gains_zoom, bins=40, color=PALETTE[3], alpha=0.7, edgecolor='#0a0a1a', linewidth=0.3)
ax.axvline(0, color='#ff6b6b', linestyle='--', linewidth=1.5, label='Break-even')
ax.set_xlabel('Listing Day Gain (%)')
ax.set_ylabel('Frequency')
ax.set_title('Listing Day Gains (Zoomed: -50% to +150%)')
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--')

fig.tight_layout()
save_fig(fig, '02_listing_gain_distribution')

# ═══════════════════════════════════════════════════════════════════════════
# 4. YEAR-WISE LISTING PERFORMANCE (Box Plot)
# ═══════════════════════════════════════════════════════════════════════════
print("4. Year-wise Listing Performance...")

fig, ax = plt.subplots(figsize=(14, 6))
df_box = df_listed[['IPO Year', 'Listing Day Gain (%)']].dropna()
df_box = df_box[df_box['Listing Day Gain (%)'].between(-100, 200)]

years_with_data = sorted(df_box['IPO Year'].unique())
data_by_year = [df_box[df_box['IPO Year'] == y]['Listing Day Gain (%)'].values for y in years_with_data]

bp = ax.boxplot(data_by_year, labels=[int(y) for y in years_with_data], patch_artist=True,
                medianprops=dict(color='#ffd93d', linewidth=2),
                whiskerprops=dict(color='#666'),
                capprops=dict(color='#666'),
                flierprops=dict(marker='o', markerfacecolor='#ff6b6b', markersize=3, alpha=0.5))
for patch, color in zip(bp['boxes'], PALETTE * 3):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(0, color='#ff6b6b', linestyle='--', linewidth=1, alpha=0.8, label='Break-even')
ax.set_xlabel('IPO Year')
ax.set_ylabel('Listing Day Gain (%)')
ax.set_title('Year-wise Listing Day Performance (Clipped to -100% to +200%)')
ax.legend()
ax.grid(axis='y', linestyle='--')
save_fig(fig, '03_yearwise_listing_boxplot')

# ═══════════════════════════════════════════════════════════════════════════
# 5. AVERAGE LISTING GAINS BY YEAR (Bar + Line)
# ═══════════════════════════════════════════════════════════════════════════
print("5. Average Listing Gains by Year...")

fig, ax = plt.subplots(figsize=(12, 5))
yearly = df_listed.groupby('IPO Year')['Listing Day Gain (%)'].agg(['mean', 'median', 'count'])
yearly.columns = ['Mean', 'Median', 'Count']

x = np.arange(len(yearly))
width = 0.35
bars1 = ax.bar(x - width/2, yearly['Mean'], width, label='Mean Gain', color=PALETTE[0], alpha=0.8, edgecolor='#0a0a1a')
bars2 = ax.bar(x + width/2, yearly['Median'], width, label='Median Gain', color=PALETTE[3], alpha=0.8, edgecolor='#0a0a1a')

ax.axhline(0, color='#ff6b6b', linestyle='--', linewidth=1)
ax.set_xlabel('IPO Year')
ax.set_ylabel('Listing Day Gain (%)')
ax.set_title('Average vs Median Listing Day Gain by Year')
ax.set_xticks(x)
ax.set_xticklabels([int(y) for y in yearly.index])
ax.legend()
ax.grid(axis='y', linestyle='--')
save_fig(fig, '04_avg_listing_gain_by_year')

# ═══════════════════════════════════════════════════════════════════════════
# 6. HIT RATE (% giving positive listing) BY YEAR
# ═══════════════════════════════════════════════════════════════════════════
print("6. Hit Rate by Year...")

fig, ax = plt.subplots(figsize=(12, 5))
hit_rate = df_listed.groupby('IPO Year')['Listing Day Gain (%)'].apply(lambda x: (x > 0).mean() * 100)
count_per_year = df_listed.groupby('IPO Year').size()

bars = ax.bar(hit_rate.index, hit_rate.values, color=PALETTE[4], alpha=0.85, edgecolor='#0a0a1a', width=0.7)
for bar, rate, cnt in zip(bars, hit_rate.values, count_per_year.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.0f}%\n(n={cnt})',
            ha='center', va='bottom', fontsize=9, color='#c084fc', fontweight='bold')

ax.axhline(50, color='#ffd93d', linestyle='--', linewidth=1, alpha=0.6, label='50% line')
ax.set_xlabel('IPO Year')
ax.set_ylabel('% IPOs with Positive Listing Gain')
ax.set_title('IPO Hit Rate by Year (% Giving Positive Listing Day Returns)')
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis='y', linestyle='--')
save_fig(fig, '05_hit_rate_by_year')

# ═══════════════════════════════════════════════════════════════════════════
# 7. SUBSCRIPTION vs LISTING GAIN (Scatter)
# ═══════════════════════════════════════════════════════════════════════════
print("7. Subscription vs Listing Gain...")

df_sub = df_listed[df_listed['Total Subscription (x)'] > 0].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Total subscription vs listing gain
ax = axes[0]
sub_vals = df_sub['Total Subscription (x)'].clip(upper=200)
gain_vals = df_sub['Listing Day Gain (%)'].clip(-100, 300)
scatter = ax.scatter(sub_vals, gain_vals, c=df_sub['IPO Year'], cmap='plasma',
                     alpha=0.6, s=25, edgecolors='#333', linewidth=0.3)
cbar = fig.colorbar(scatter, ax=ax, label='Year')
ax.axhline(0, color='#ff6b6b', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Total Subscription (x) [clipped at 200x]')
ax.set_ylabel('Listing Day Gain (%)')
ax.set_title('Total Subscription vs Listing Day Gain')
ax.grid(linestyle='--')

# RII subscription vs listing gain
ax = axes[1]
df_rii = df_sub[df_sub['RII Subscription (x)'] > 0]
rii_vals = df_rii['RII Subscription (x)'].clip(upper=100)
gain_vals2 = df_rii['Listing Day Gain (%)'].clip(-100, 300)
ax.scatter(rii_vals, gain_vals2, c=PALETTE[1], alpha=0.5, s=25, edgecolors='#333', linewidth=0.3)
ax.axhline(0, color='#ff6b6b', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('RII (Retail) Subscription (x) [clipped at 100x]')
ax.set_ylabel('Listing Day Gain (%)')
ax.set_title('Retail Subscription vs Listing Day Gain')
ax.grid(linestyle='--')

fig.tight_layout()
save_fig(fig, '06_subscription_vs_listing_gain')

# ═══════════════════════════════════════════════════════════════════════════
# 8. LISTING DAY vs LISTING WEEK (Scatter)
# ═══════════════════════════════════════════════════════════════════════════
print("8. Listing Day vs Listing Week Returns...")

df_both = df_listed.dropna(subset=['Listing Day Gain (%)', 'Listing Week Gain (%)']).copy()

fig, ax = plt.subplots(figsize=(8, 7))
d = df_both['Listing Day Gain (%)'].clip(-100, 200)
w = df_both['Listing Week Gain (%)'].clip(-100, 200)

ax.scatter(d, w, c=PALETTE[0], alpha=0.4, s=20, edgecolors='#333', linewidth=0.2)
# 45 degree line
lims = [-100, 200]
ax.plot(lims, lims, color='#ffd93d', linestyle='--', linewidth=1, alpha=0.7, label='Same returns')
ax.axhline(0, color='#ff6b6b', linestyle=':', linewidth=0.8)
ax.axvline(0, color='#ff6b6b', linestyle=':', linewidth=0.8)
ax.set_xlabel('Listing Day Gain (%)')
ax.set_ylabel('Listing Week Gain (%)')
ax.set_title('Listing Day vs Listing Week Returns')
ax.legend()
ax.grid(linestyle='--')
save_fig(fig, '07_day_vs_week_returns')

# ═══════════════════════════════════════════════════════════════════════════
# 9. TOP 20 BEST & WORST IPOs (Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════════════
print("9. Top/Bottom IPOs...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 20
ax = axes[0]
top20 = df_listed.nlargest(20, 'Listing Day Gain (%)')
labels = [f"{row['Symbol']} ({int(row['IPO Year'])})" for _, row in top20.iterrows()]
vals = top20['Listing Day Gain (%)'].values
bars = ax.barh(range(len(vals)), vals, color=PALETTE[3], alpha=0.85, edgecolor='#0a0a1a')
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Listing Day Gain (%)')
ax.set_title('Top 20 Best Listing Day Gains')
ax.invert_yaxis()
ax.grid(axis='x', linestyle='--')
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f'+{v:.0f}%',
            va='center', fontsize=8, color='#6bcb77')

# Bottom 20
ax = axes[1]
bot20 = df_listed.nsmallest(20, 'Listing Day Gain (%)')
labels = [f"{row['Symbol']} ({int(row['IPO Year'])})" for _, row in bot20.iterrows()]
vals = bot20['Listing Day Gain (%)'].values
bars = ax.barh(range(len(vals)), vals, color=PALETTE[1], alpha=0.85, edgecolor='#0a0a1a')
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Listing Day Gain (%)')
ax.set_title('Top 20 Worst Listing Day Gains')
ax.invert_yaxis()
ax.grid(axis='x', linestyle='--')
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() - 2, bar.get_y() + bar.get_height()/2, f'{v:.0f}%',
            va='center', ha='right', fontsize=8, color='#ff6b6b')

fig.tight_layout()
save_fig(fig, '08_top_bottom_ipos')

# ═══════════════════════════════════════════════════════════════════════════
# 10. SECURITY TYPE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("10. Security Type Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count by type
ax = axes[0]
type_counts = df_listed.groupby('Security Type').size().sort_values(ascending=False)
bars = ax.bar(type_counts.index, type_counts.values, color=PALETTE[:len(type_counts)], alpha=0.85, edgecolor='#0a0a1a')
for bar, c in zip(bars, type_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(c),
            ha='center', fontsize=10, fontweight='bold', color='#e0e0e0')
ax.set_xlabel('Security Type')
ax.set_ylabel('Count')
ax.set_title('IPO Count by Security Type')
ax.grid(axis='y', linestyle='--')

# Mean listing gain by type
ax = axes[1]
type_gain = df_listed.groupby('Security Type')['Listing Day Gain (%)'].mean().sort_values()
colors_bar = [PALETTE[1] if v < 0 else PALETTE[3] for v in type_gain.values]
bars = ax.barh(type_gain.index, type_gain.values, color=colors_bar, alpha=0.85, edgecolor='#0a0a1a')
ax.axvline(0, color='#ffd93d', linestyle='--', linewidth=1)
ax.set_xlabel('Mean Listing Day Gain (%)')
ax.set_title('Avg Listing Day Gain by Security Type')
ax.grid(axis='x', linestyle='--')

fig.tight_layout()
save_fig(fig, '09_security_type_analysis')

# ═══════════════════════════════════════════════════════════════════════════
# 11. SUBSCRIPTION PATTERNS (QIB vs NII vs RII)
# ═══════════════════════════════════════════════════════════════════════════
print("11. Subscription Patterns...")

fig, ax = plt.subplots(figsize=(14, 6))
df_sub_year = df_listed[df_listed['Total Subscription (x)'] > 0].copy()
sub_cols = ['QIB Subscription (x)', 'NII Subscription (x)', 'RII Subscription (x)']
sub_yearly = df_sub_year.groupby('IPO Year')[sub_cols].median()

x = np.arange(len(sub_yearly))
width = 0.25
for i, (col, color) in enumerate(zip(sub_cols, [PALETTE[0], PALETTE[1], PALETTE[2]])):
    label = col.replace(' Subscription (x)', '')
    ax.bar(x + i*width, sub_yearly[col].clip(upper=100), width, label=label, color=color, alpha=0.8, edgecolor='#0a0a1a')

ax.set_xlabel('IPO Year')
ax.set_ylabel('Median Subscription (x) [clipped at 100x]')
ax.set_title('Median Subscription Levels by Investor Category (Year-wise)')
ax.set_xticks(x + width)
ax.set_xticklabels([int(y) for y in sub_yearly.index])
ax.legend()
ax.grid(axis='y', linestyle='--')
save_fig(fig, '10_subscription_patterns')

# ═══════════════════════════════════════════════════════════════════════════
# 12. ISSUE PRICE RANGE vs LISTING PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
print("12. Issue Price vs Listing Performance...")

df_price = df_listed[df_listed['Issue Price (₹)'].notna() & (df_listed['Issue Price (₹)'] > 0)].copy()

# Create price buckets
bins = [0, 50, 100, 200, 500, 1000, 2500, float('inf')]
labels_price = ['₹0-50', '₹50-100', '₹100-200', '₹200-500', '₹500-1000', '₹1000-2500', '₹2500+']
df_price['Price Bucket'] = pd.cut(df_price['Issue Price (₹)'], bins=bins, labels=labels_price)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Count per bucket
ax = axes[0]
bucket_counts = df_price.groupby('Price Bucket', observed=True).size()
bars = ax.bar(range(len(bucket_counts)), bucket_counts.values, color=PALETTE[:len(bucket_counts)], alpha=0.85, edgecolor='#0a0a1a')
ax.set_xticks(range(len(bucket_counts)))
ax.set_xticklabels(bucket_counts.index, rotation=30)
for bar, c in zip(bars, bucket_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(c),
            ha='center', fontsize=9, fontweight='bold', color='#e0e0e0')
ax.set_ylabel('Number of IPOs')
ax.set_title('IPOs by Issue Price Range')
ax.grid(axis='y', linestyle='--')

# Mean gain per bucket
ax = axes[1]
bucket_gain = df_price.groupby('Price Bucket', observed=True)['Listing Day Gain (%)'].agg(['mean', 'median'])
x_pos = range(len(bucket_gain))
ax.bar([p - 0.15 for p in x_pos], bucket_gain['mean'], 0.3, label='Mean', color=PALETTE[0], alpha=0.8, edgecolor='#0a0a1a')
ax.bar([p + 0.15 for p in x_pos], bucket_gain['median'], 0.3, label='Median', color=PALETTE[3], alpha=0.8, edgecolor='#0a0a1a')
ax.set_xticks(list(x_pos))
ax.set_xticklabels(bucket_gain.index, rotation=30)
ax.axhline(0, color='#ff6b6b', linestyle='--', linewidth=1)
ax.set_ylabel('Listing Day Gain (%)')
ax.set_title('Listing Day Gain by Issue Price Range')
ax.legend()
ax.grid(axis='y', linestyle='--')

fig.tight_layout()
save_fig(fig, '11_price_range_analysis')

# ═══════════════════════════════════════════════════════════════════════════
# 13. CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
print("13. Correlation Heatmap...")

corr_cols = ['Issue Price (₹)', 'QIB Subscription (x)', 'NII Subscription (x)',
             'RII Subscription (x)', 'Total Subscription (x)',
             'Listing Day Gain (%)', 'Listing Week Gain (%)']

df_corr = df_listed[corr_cols].dropna()
corr = df_corr.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax, linewidths=0.5, linecolor='#0f0f1a',
            annot_kws={'fontsize': 9},
            xticklabels=[c.replace(' (₹)', '').replace(' (x)', '').replace(' (%)', '') for c in corr.columns],
            yticklabels=[c.replace(' (₹)', '').replace(' (x)', '').replace(' (%)', '') for c in corr.columns])
ax.set_title('Correlation Matrix of Key IPO Metrics')
ax.tick_params(labelsize=9)
fig.tight_layout()
save_fig(fig, '12_correlation_heatmap')

# ═══════════════════════════════════════════════════════════════════════════
# 14. MONTHLY SEASONALITY
# ═══════════════════════════════════════════════════════════════════════════
print("14. Monthly Seasonality...")

df_monthly = df_listed[df_listed['IPO Start Date'].notna()].copy()
df_monthly['Month'] = df_monthly['IPO Start Date'].dt.month

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# IPOs per month
ax = axes[0]
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_counts = df_monthly.groupby('Month').size().reindex(range(1, 13), fill_value=0)
ax.bar(range(1, 13), month_counts.values, color=PALETTE[4], alpha=0.85, edgecolor='#0a0a1a')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_ylabel('Number of IPOs')
ax.set_title('IPO Activity by Month')
ax.grid(axis='y', linestyle='--')

# Avg gain by month
ax = axes[1]
month_gain = df_monthly.groupby('Month')['Listing Day Gain (%)'].median().reindex(range(1, 13), fill_value=0)
colors_month = [PALETTE[3] if v >= 0 else PALETTE[1] for v in month_gain.values]
ax.bar(range(1, 13), month_gain.values, color=colors_month, alpha=0.85, edgecolor='#0a0a1a')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.axhline(0, color='#ffd93d', linestyle='--', linewidth=1)
ax.set_ylabel('Median Listing Day Gain (%)')
ax.set_title('Median Listing Day Gain by Month')
ax.grid(axis='y', linestyle='--')

fig.tight_layout()
save_fig(fig, '13_monthly_seasonality')

# ═══════════════════════════════════════════════════════════════════════════
# 15. EQ vs SME COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("15. EQ vs SME Comparison...")

df_listed['Category'] = df_listed['Security Type'].apply(lambda x: 'Mainboard (EQ)' if x == 'EQ' else 'SME')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Count
ax = axes[0]
cat_counts = df_listed.groupby('Category').size()
ax.bar(cat_counts.index, cat_counts.values, color=[PALETTE[0], PALETTE[2]], alpha=0.85, edgecolor='#0a0a1a')
for i, (cat, cnt) in enumerate(cat_counts.items()):
    ax.text(i, cnt + 1, str(cnt), ha='center', fontweight='bold', fontsize=12, color='#e0e0e0')
ax.set_ylabel('Count')
ax.set_title('IPO Count: Mainboard vs SME')
ax.grid(axis='y', linestyle='--')

# Mean/Median gain
ax = axes[1]
cat_gain = df_listed.groupby('Category')['Listing Day Gain (%)'].agg(['mean', 'median'])
x = np.arange(len(cat_gain))
ax.bar(x - 0.15, cat_gain['mean'], 0.3, label='Mean', color=PALETTE[0], alpha=0.8, edgecolor='#0a0a1a')
ax.bar(x + 0.15, cat_gain['median'], 0.3, label='Median', color=PALETTE[3], alpha=0.8, edgecolor='#0a0a1a')
ax.set_xticks(x)
ax.set_xticklabels(cat_gain.index)
ax.axhline(0, color='#ff6b6b', linestyle='--', linewidth=1)
ax.set_ylabel('Listing Day Gain (%)')
ax.set_title('Avg Listing Gain: Mainboard vs SME')
ax.legend()
ax.grid(axis='y', linestyle='--')

# Hit rate
ax = axes[2]
cat_hit = df_listed.groupby('Category')['Listing Day Gain (%)'].apply(lambda x: (x > 0).mean() * 100)
bars = ax.bar(cat_hit.index, cat_hit.values, color=[PALETTE[0], PALETTE[2]], alpha=0.85, edgecolor='#0a0a1a')
for bar, v in zip(bars, cat_hit.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{v:.1f}%',
            ha='center', fontweight='bold', fontsize=12, color='#e0e0e0')
ax.axhline(50, color='#ffd93d', linestyle='--', linewidth=1, alpha=0.6)
ax.set_ylabel('% Positive Listing')
ax.set_title('Hit Rate: Mainboard vs SME')
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--')

fig.tight_layout()
save_fig(fig, '14_eq_vs_sme')

# ═══════════════════════════════════════════════════════════════════════════
# 16. CUMULATIVE LISTING GAINS OVER TIME
# ═══════════════════════════════════════════════════════════════════════════
print("16. Cumulative Returns if investing in every IPO at issue price...")

df_time = df_listed[df_listed['Listing Date'].notna()].sort_values('Listing Date').copy()
df_time['Cumulative Mean Gain'] = df_time['Listing Day Gain (%)'].expanding().mean()
df_time['Rolling 20 Mean Gain'] = df_time['Listing Day Gain (%)'].rolling(20, min_periods=5).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_time['Listing Date'], df_time['Cumulative Mean Gain'], color=PALETTE[0], linewidth=1.5, label='Cumulative Avg Gain', alpha=0.8)
ax.plot(df_time['Listing Date'], df_time['Rolling 20 Mean Gain'], color=PALETTE[2], linewidth=1.5, label='Rolling 20-IPO Avg Gain', alpha=0.8)
ax.axhline(0, color='#ff6b6b', linestyle='--', linewidth=1, alpha=0.6)
ax.fill_between(df_time['Listing Date'], df_time['Rolling 20 Mean Gain'], 0,
                where=df_time['Rolling 20 Mean Gain'] > 0, color=PALETTE[3], alpha=0.1)
ax.fill_between(df_time['Listing Date'], df_time['Rolling 20 Mean Gain'], 0,
                where=df_time['Rolling 20 Mean Gain'] <= 0, color=PALETTE[1], alpha=0.1)
ax.set_xlabel('Listing Date')
ax.set_ylabel('Average Listing Day Gain (%)')
ax.set_title('Average Listing Day Gains Over Time')
ax.legend()
ax.grid(linestyle='--')
save_fig(fig, '15_cumulative_gains_over_time')

# ═══════════════════════════════════════════════════════════════════════════
# 17. REGISTRAR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("17. Registrar Analysis...")

df_reg = df_listed[df_listed['Registrar'].notna()].copy()
# Clean registrar names
df_reg['Registrar_Clean'] = df_reg['Registrar'].str.strip()
df_reg['Registrar_Clean'] = df_reg['Registrar_Clean'].str.replace(r',.*', '', regex=True)

reg_stats = df_reg.groupby('Registrar_Clean').agg(
    count=('Listing Day Gain (%)', 'size'),
    mean_gain=('Listing Day Gain (%)', 'mean'),
    median_gain=('Listing Day Gain (%)', 'median'),
    hit_rate=('Listing Day Gain (%)', lambda x: (x > 0).mean() * 100)
).sort_values('count', ascending=False).head(8)

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(reg_stats))
ax.bar(x - 0.2, reg_stats['mean_gain'], 0.4, label='Mean Gain (%)', color=PALETTE[0], alpha=0.8, edgecolor='#0a0a1a')
ax.bar(x + 0.2, reg_stats['hit_rate'], 0.4, label='Hit Rate (%)', color=PALETTE[3], alpha=0.8, edgecolor='#0a0a1a')
ax.set_xticks(x)
ax.set_xticklabels([r[:20] for r in reg_stats.index], rotation=30, ha='right', fontsize=9)
ax.axhline(0, color='#ff6b6b', linestyle='--', linewidth=1)
ax.set_title('Top Registrars: Mean Listing Gain & Hit Rate')
ax.legend()
ax.grid(axis='y', linestyle='--')
# Add count labels
for i, cnt in enumerate(reg_stats['count']):
    ax.text(i, max(reg_stats['mean_gain'].iloc[i], reg_stats['hit_rate'].iloc[i]) + 3,
            f'n={cnt}', ha='center', fontsize=8, color='#b0b0b0')

fig.tight_layout()
save_fig(fig, '16_registrar_analysis')

# ═══════════════════════════════════════════════════════════════════════════
# 18. PRINT SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

report = {
    "Total IPOs (Equity/SME)": len(df_equity),
    "IPOs with Listing Data": len(df_listed),
    "Years Covered": f"{int(df_equity['IPO Year'].min())} - {int(df_equity['IPO Year'].max())}",
    "Overall Mean Listing Day Gain": f"{lg.mean():.2f}%",
    "Overall Median Listing Day Gain": f"{lg.median():.2f}%",
    "Hit Rate (% positive listing)": f"{(lg > 0).mean()*100:.1f}%",
    "Best IPO": stats['best_ipo'],
    "Worst IPO": stats['worst_ipo'],
    "Mainboard IPOs": len(df_listed[df_listed['Category'] == 'Mainboard (EQ)']),
    "SME IPOs": len(df_listed[df_listed['Category'] != 'Mainboard (EQ)']),
    "Mainboard Avg Gain": f"{df_listed[df_listed['Category']=='Mainboard (EQ)']['Listing Day Gain (%)'].mean():.2f}%",
    "SME Avg Gain": f"{df_listed[df_listed['Category']!='Mainboard (EQ)']['Listing Day Gain (%)'].mean():.2f}%",
}

for k, v in report.items():
    print(f"  {k}: {v}")

# Year-wise summary table
print("\nYear-wise Summary:")
yearly_summary = df_listed.groupby('IPO Year').agg(
    Count=('Listing Day Gain (%)', 'size'),
    Mean_Gain=('Listing Day Gain (%)', 'mean'),
    Median_Gain=('Listing Day Gain (%)', 'median'),
    Hit_Rate=('Listing Day Gain (%)', lambda x: f"{(x > 0).mean()*100:.0f}%"),
    Best=('Listing Day Gain (%)', 'max'),
    Worst=('Listing Day Gain (%)', 'min'),
)
print(yearly_summary.to_string())

# Save report as JSON
with open(OUTPUT_DIR / 'analysis_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✅ All charts saved to: {OUTPUT_DIR}")
print(f"  Total charts: 16")
print("Done!")