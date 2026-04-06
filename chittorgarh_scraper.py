import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

all_data = []

years = range(2016, 2027)  # 2016 to 2026

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

COLUMNS = ["Company", "Listing Day Gain / Loss", "Current Gain / Loss", "Year"]

for year in years:
    print(f"Fetching {year}...")
    url = f"https://www.chittorgarh.com/ipo/ipo_perf_tracker.asp?year={year}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Request failed for {year}: {e}")
        continue

    soup = BeautifulSoup(response.text, "html.parser")

    # The page has two tables; the second one contains the actual IPO data.
    tables = soup.find_all("table")
    if len(tables) < 2:
        print(f"  Expected 2 tables but found {len(tables)} for {year}, skipping.")
        continue

    table = tables[1]  # index 1 = real data table
    rows = table.find_all("tr")

    if len(rows) <= 1:
        print(f"  No data rows found for {year}")
        continue

    added = 0
    for row in rows[1:]:  # skip header row
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cols) == 3:  # Company, Listing Day Gain, Current Gain
            cols.append(year)
            all_data.append(cols)
            added += 1

    print(f"  Added {added} IPOs for {year}")
    time.sleep(1)  # be polite, don't hammer the server

# Save to Excel
if all_data:
    df = pd.DataFrame(all_data, columns=COLUMNS)
    output_file = BASE_DIR / "ipo_listing_prices_2016_2026.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\nDone! Saved {len(df)} records to {output_file}")
else:
    print("\nNo data collected. Nothing saved.")