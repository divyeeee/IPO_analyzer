"""
NSE India IPO Scraper
=====================
Scrapes IPO issue details and bid/subscription details from NSE India's
internal APIs using Selenium (headless Chrome) to handle Akamai bot protection.

Output:
  - nse_ipo_issue_details.xlsx  (one row per IPO)
  - nse_ipo_bid_details.xlsx    (one row per category per IPO)

Usage:
  python nse_ipo_scraper.py
"""

import json
import time
import sys
from datetime import datetime

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException

# ─── Configuration ───────────────────────────────────────────────────────────

START_YEAR = 2016
COOKIE_REFRESH_EVERY = 40       # re-visit homepage every N requests
REQUEST_DELAY = 1.5             # seconds between API calls
MAX_RETRIES = 3                 # retries per IPO on failure
ISSUE_DETAILS_FILE = "nse_ipo_issue_details.xlsx"
BID_DETAILS_FILE = "nse_ipo_bid_details.xlsx"

NSE_HOME = "https://www.nseindia.com/"
PAST_ISSUES_URL = "https://www.nseindia.com/api/public-past-issues"
IPO_DETAIL_URL = "https://www.nseindia.com/api/ipo-detail?symbol={symbol}&series=EQ"

# Issue-info fields we want to extract (title → column name mapping)
ISSUE_FIELDS_MAP = {
    "Issue Period": "Issue Period",
    "Issue Type": "Issue Type",
    "Issue Size": "Issue Size",
    "Issue Size (in Crores)": "Issue Size",
    "Price Range": "Detail Price Range",
    "Discount": "Discount",
    "Face Value": "Face Value",
    "Lot Size": "Lot Size",
    "Fresh Issue Size": "Fresh Issue Size",
    "Fresh Issue Size (in Crores)": "Fresh Issue Size",
    "OFS Issue Size": "OFS Issue Size",
    "OFS Issue Size (in Crores)": "OFS Issue Size",
    "Total Issue Size": "Total Issue Size",
    "Total Issue Size (in Crores)": "Total Issue Size",
    "Listing At": "Listing At",
    "Categories applicable": "Categories",
    "Sub-Categories applicable for UPI": "UPI Sub-Categories",
    "Name of the Registrar": "Registrar",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def create_driver():
    """Create a headless Chrome driver with anti-detection settings."""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=opts)
    # Hide webdriver flag
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'},
    )
    return driver


def refresh_cookies(driver):
    """Visit NSE homepage to get fresh session cookies."""
    print("  🔄  Refreshing NSE session cookies...")
    driver.get(NSE_HOME)
    time.sleep(3)


def fetch_json(driver, url, retries=MAX_RETRIES):
    """Navigate to a JSON API URL and parse the response body."""
    for attempt in range(1, retries + 1):
        try:
            driver.get(url)
            time.sleep(REQUEST_DELAY)
            body = driver.find_element(By.TAG_NAME, "body").text
            if not body or body.strip().startswith("<"):
                raise ValueError("Non-JSON response")
            return json.loads(body)
        except (json.JSONDecodeError, ValueError, WebDriverException) as exc:
            if attempt < retries:
                print(f"    ⚠  Attempt {attempt} failed ({exc}), refreshing cookies...")
                refresh_cookies(driver)
            else:
                raise


def parse_ipo_date(date_str):
    """Parse NSE date string like '25-MAR-2026' into a datetime, or None."""
    if not date_str or date_str == "-":
        return None
    for fmt in ("%d-%b-%Y", "%d-%B-%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def extract_issue_info(issue_info_list):
    """Extract key fields from issueInfo list of {title, value} dicts."""
    extracted = {}
    for item in issue_info_list:
        title = (item.get("title") or "").strip()
        value = (item.get("value") or "").strip().strip('"')
        if title in ISSUE_FIELDS_MAP:
            col = ISSUE_FIELDS_MAP[title]
            # Only keep first occurrence for duplicate column mappings
            if col not in extracted:
                extracted[col] = value
    return extracted


def extract_bid_details(bid_details_list, symbol, company_name):
    """Convert bidDetails list into rows for the bid DataFrame."""
    rows = []
    for bid in bid_details_list:
        category = bid.get("category", "")
        # Skip header rows
        if category == "Category" or not category:
            continue
        rows.append({
            "Symbol": symbol,
            "Company Name": company_name,
            "Category": category,
            "Shares Offered": bid.get("noOfSharesOffered", ""),
            "Shares Bid For": bid.get("noOfsharesBid", ""),
            "Subscription (x)": bid.get("noOfTime", ""),
        })
    return rows


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NSE India IPO Scraper")
    print(f"  Fetching IPOs from {START_YEAR} to present")
    print("=" * 60)

    driver = create_driver()
    try:
        # Step 1: Get initial cookies
        refresh_cookies(driver)

        # Step 2: Fetch all past IPOs
        print("\n📥  Fetching master list of past IPOs...")
        past_issues = fetch_json(driver, PAST_ISSUES_URL)
        print(f"    Total past IPOs on NSE: {len(past_issues)}")

        # Step 3: Filter to START_YEAR onwards
        filtered = []
        for ipo in past_issues:
            dt = parse_ipo_date(ipo.get("ipoStartDate", ""))
            if dt and dt.year >= START_YEAR:
                filtered.append(ipo)
        filtered.sort(key=lambda x: parse_ipo_date(x.get("ipoStartDate", "")) or datetime.min)
        print(f"    IPOs from {START_YEAR} onwards: {len(filtered)}\n")

        issue_rows = []
        bid_rows = []
        failed_symbols = []
        request_count = 0

        # Step 4: Fetch details for each IPO
        for idx, ipo in enumerate(filtered, 1):
            symbol = ipo["symbol"]
            company = ipo.get("companyName", ipo.get("company", symbol))
            sec_type = ipo.get("securityType", "")

            print(f"[{idx}/{len(filtered)}] {company} ({symbol}) [{sec_type}]")

            # Refresh cookies periodically
            if request_count > 0 and request_count % COOKIE_REFRESH_EVERY == 0:
                refresh_cookies(driver)

            # Determine series (SME often uses "SM", equity uses "EQ")
            series = "SM" if sec_type == "SME" else "EQ"
            detail_url = f"https://www.nseindia.com/api/ipo-detail?symbol={symbol}&series={series}"

            try:
                detail = fetch_json(driver, detail_url)
                request_count += 1
            except Exception as exc:
                print(f"    ❌  Failed to fetch detail: {exc}")
                # Try alternate series
                alt_series = "EQ" if series == "SM" else "SM"
                alt_url = f"https://www.nseindia.com/api/ipo-detail?symbol={symbol}&series={alt_series}"
                try:
                    detail = fetch_json(driver, alt_url)
                    request_count += 1
                except Exception:
                    print(f"    ❌  Also failed with series={alt_series}, skipping.")
                    failed_symbols.append(symbol)
                    continue

            # ── Issue Details ──
            row = {
                "Symbol": symbol,
                "Company Name": company,
                "Security Type": sec_type,
                "IPO Start Date": ipo.get("ipoStartDate", ""),
                "IPO End Date": ipo.get("ipoEndDate", ""),
                "Price Range": ipo.get("priceRange", ""),
                "Issue Price": ipo.get("issuePrice", ""),
                "Listing Date": ipo.get("listingDate", ""),
            }

            # Extract from issueInfo — API returns dict with "dataList" key
            raw_issue = detail.get("issueInfo", {})
            if isinstance(raw_issue, dict):
                issue_info = raw_issue.get("dataList", raw_issue.get("data", []))
            elif isinstance(raw_issue, list):
                issue_info = raw_issue
            else:
                issue_info = []

            if isinstance(issue_info, list):
                row.update(extract_issue_info(issue_info))

            issue_rows.append(row)

            # ── Bid Details ──
            bid_details = detail.get("bidDetails", [])
            if isinstance(bid_details, list):
                bid_rows.extend(extract_bid_details(bid_details, symbol, company))

        # Step 5: Save to Excel
        print("\n" + "=" * 60)
        print("📊  Saving results...")

        if issue_rows:
            df_issue = pd.DataFrame(issue_rows)
            df_issue.to_excel(ISSUE_DETAILS_FILE, index=False)
            print(f"  ✅  {ISSUE_DETAILS_FILE}: {len(df_issue)} IPOs")

        if bid_rows:
            df_bid = pd.DataFrame(bid_rows)
            df_bid.to_excel(BID_DETAILS_FILE, index=False)
            print(f"  ✅  {BID_DETAILS_FILE}: {len(df_bid)} bid rows")

        if failed_symbols:
            print(f"\n  ⚠  Failed to fetch {len(failed_symbols)} IPOs: {', '.join(failed_symbols)}")

        print("\nDone! 🎉")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
