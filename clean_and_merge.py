"""
Clean and merge NSE IPO issue details and bid details into a single CSV.

Reads:
  - nse_ipo_issue_details.xlsx
  - nse_ipo_bid_details.xlsx

Produces:
  - nse_ipo_merged.csv  (one row per IPO with aggregated bid subscription data)
"""

import pandas as pd
import re
import sys

# ─── Read data ───────────────────────────────────────────────────────────────

print("📖  Reading Excel files...")
df_issue = pd.read_excel("nse_ipo_issue_details.xlsx", engine="openpyxl")
df_bid = pd.read_excel("nse_ipo_bid_details.xlsx", engine="openpyxl")

print(f"  Issue details: {len(df_issue)} rows, {len(df_issue.columns)} cols")
print(f"  Bid details:   {len(df_bid)} rows, {len(df_bid.columns)} cols")


# ─── Clean issue details ─────────────────────────────────────────────────────

print("\n🧹  Cleaning issue details...")

# Replace "-" and empty strings with NaN
df_issue.replace(["-", "", "nan"], pd.NA, inplace=True)

# Parse Issue Price: extract numeric value (e.g., "Rs.390" → 390)
def parse_price(val):
    if pd.isna(val):
        return pd.NA
    val = str(val).replace(",", "")
    # Try to extract a number (handles "Rs.390", "390", "390.00", etc.)
    match = re.search(r"(\d+(?:\.\d+)?)", val)
    return float(match.group(1)) if match else pd.NA

df_issue["Issue Price (₹)"] = df_issue["Issue Price"].apply(parse_price)

# Parse Price Range: extract low and high
def parse_price_range(val):
    if pd.isna(val):
        return pd.NA, pd.NA
    val = str(val).replace(",", "")
    nums = re.findall(r"(\d+(?:\.\d+)?)", val)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[-1])
    elif len(nums) == 1:
        return float(nums[0]), float(nums[0])
    return pd.NA, pd.NA

df_issue[["Price Band Low (₹)", "Price Band High (₹)"]] = (
    df_issue["Price Range"].apply(lambda x: pd.Series(parse_price_range(x)))
)

# Parse Face Value
df_issue["Face Value (₹)"] = df_issue.get("Face Value", pd.Series(dtype=float)).apply(parse_price)

# Parse Lot Size
def parse_lot(val):
    if pd.isna(val):
        return pd.NA
    match = re.search(r"(\d+)", str(val).replace(",", ""))
    return int(match.group(1)) if match else pd.NA

df_issue["Lot Size (shares)"] = df_issue.get("Lot Size", pd.Series(dtype=float)).apply(parse_lot)

# Parse dates
def parse_nse_date(val):
    if pd.isna(val):
        return pd.NaT
    val = str(val).strip()
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return pd.Timestamp(pd.to_datetime(val, format=fmt))
        except (ValueError, TypeError):
            continue
    try:
        return pd.Timestamp(pd.to_datetime(val))
    except Exception:
        return pd.NaT

for col in ["IPO Start Date", "IPO End Date", "Listing Date"]:
    if col in df_issue.columns:
        df_issue[col + " (parsed)"] = df_issue[col].apply(parse_nse_date)

# Year column
if "IPO Start Date (parsed)" in df_issue.columns:
    df_issue["IPO Year"] = df_issue["IPO Start Date (parsed)"].dt.year

print(f"  Cleaned {len(df_issue)} issue rows")


# ─── Clean bid details ──────────────────────────────────────────────────────

print("🧹  Cleaning bid details...")

df_bid.replace(["-", "", "nan"], pd.NA, inplace=True)

# Parse numeric bid columns
for col in ["Shares Offered", "Shares Bid For", "Subscription (x)"]:
    if col in df_bid.columns:
        df_bid[col] = pd.to_numeric(
            df_bid[col].astype(str).str.replace(",", ""), errors="coerce"
        )

# Aggregate: pivot key categories into columns per IPO
# Main categories: QIB, NII, RII (Retail), Employee, Total
def categorize(cat_str):
    if pd.isna(cat_str):
        return "Other"
    c = str(cat_str).lower()
    if "total" in c:
        return "Total"
    if "qualified institutional" in c or "qib" in c:
        # Only the main QIB row, not sub-categories
        if any(x in c for x in ["foreign", "domestic", "mutual", "others", "insurance"]):
            return None  # sub-category, skip for aggregation
        return "QIB"
    if "non institutional" in c or "nii" in c:
        if "more than" in c or "less than" in c:
            return None  # sub-category
        return "NII"
    if "retail" in c or "rii" in c:
        return "RII"
    if "employee" in c:
        return "Employee"
    return None

df_bid["Main Category"] = df_bid["Category"].apply(categorize)

# Keep only main categories
df_bid_main = df_bid[df_bid["Main Category"].notna()].copy()

print(f"  Main category rows: {len(df_bid_main)}")
print(f"  Categories: {df_bid_main['Main Category'].value_counts().to_dict()}")


# ─── Pivot bid data: one row per IPO ────────────────────────────────────────

print("\n🔗  Pivoting bid details...")

# Pivot subscription ratios
sub_pivot = df_bid_main.pivot_table(
    index="Symbol",
    columns="Main Category",
    values="Subscription (x)",
    aggfunc="first",
)
sub_pivot.columns = [f"{c} Subscription (x)" for c in sub_pivot.columns]

# Pivot shares offered
offered_pivot = df_bid_main.pivot_table(
    index="Symbol",
    columns="Main Category",
    values="Shares Offered",
    aggfunc="first",
)
offered_pivot.columns = [f"{c} Shares Offered" for c in offered_pivot.columns]

# Pivot shares bid for
bid_pivot = df_bid_main.pivot_table(
    index="Symbol",
    columns="Main Category",
    values="Shares Bid For",
    aggfunc="first",
)
bid_pivot.columns = [f"{c} Shares Bid For" for c in bid_pivot.columns]

# Merge all pivots
bid_agg = sub_pivot.join(offered_pivot, how="outer").join(bid_pivot, how="outer")
bid_agg = bid_agg.reset_index()

print(f"  Pivoted bid data: {len(bid_agg)} IPOs with subscription data")


# ─── Merge issue + bid ──────────────────────────────────────────────────────

print("🔗  Merging issue details with bid data...")

df_merged = df_issue.merge(bid_agg, on="Symbol", how="left")

# Select and order final columns
final_cols = [
    "Symbol",
    "Company Name",
    "Security Type",
    "IPO Year",
    "IPO Start Date",
    "IPO End Date",
    "Listing Date",
    "Issue Price (₹)",
    "Price Band Low (₹)",
    "Price Band High (₹)",
    "Face Value (₹)",
    "Lot Size (shares)",
    "Issue Size",
    "Issue Type",
    "Listing At",
    "Categories",
    "Registrar",
    "QIB Subscription (x)",
    "NII Subscription (x)",
    "RII Subscription (x)",
    "Employee Subscription (x)",
    "Total Subscription (x)",
    "QIB Shares Offered",
    "NII Shares Offered",
    "RII Shares Offered",
    "Total Shares Offered",
    "QIB Shares Bid For",
    "NII Shares Bid For",
    "RII Shares Bid For",
    "Total Shares Bid For",
]

# Only keep columns that exist
final_cols = [c for c in final_cols if c in df_merged.columns]
df_final = df_merged[final_cols].copy()

# Sort by IPO date descending
if "IPO Start Date" in df_final.columns:
    df_final = df_final.sort_values("IPO Year", ascending=True, na_position="last")

# ─── Save ────────────────────────────────────────────────────────────────────

output_file = "nse_ipo_merged.csv"
df_final.to_csv(output_file, index=False)

print(f"\n✅  Saved {len(df_final)} rows × {len(df_final.columns)} cols to {output_file}")
print(f"\nColumn summary:")
for col in df_final.columns:
    non_null = df_final[col].notna().sum()
    print(f"  {col}: {non_null}/{len(df_final)} non-null")

print("\nDone! 🎉")
