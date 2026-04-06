import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# ─── Config ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "nse_ipo_merged.csv"
OUTPUT_CSV = BASE_DIR / "nse_ipo_merged.csv"
REQUEST_DELAY = 1.5
COOKIE_REFRESH = 40
NSE_HOME = "https://www.nseindia.com/"

# ─── Setup Browser ──────────────────────────────────────────────────────────
opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--disable-blink-features=AutomationControlled")
opts.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

def create_driver():
    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'},
    )
    return driver

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    print("📖 Reading merged IPO data...")
    df = pd.read_csv(INPUT_CSV)
    
    # Add new columns
    if "Listing Day Close (₹)" not in df.columns:
        df["Listing Day Close (₹)"] = pd.NA
        df["Listing Week Close (₹)"] = pd.NA
        df["Listing Day Gain (%)"] = pd.NA
        df["Listing Week Gain (%)"] = pd.NA

    df["Listing Date"] = pd.to_datetime(df["Listing Date"], errors="coerce")
    
    # Filter usable
    usable_idx = df[df["Listing Date"].notna() & df["Issue Price (₹)"].notna()].index
    print(f"🎯 Total usable IPOs (with Listing Date and Issue Price): {len(usable_idx)}")

    driver = create_driver()
    try:
        print("🔄 Getting initial session cookies...")
        driver.get(NSE_HOME)
        time.sleep(3)

        success = 0
        req_count = 0
        
        for idx in usable_idx:
            # Skip if already computed
            if pd.notna(df.at[idx, "Listing Day Gain (%)"]):
                continue
                
            symbol = df.at[idx, "Symbol"]
            series = "SM" if df.at[idx, "Security Type"] == "SME" else "EQ"
            list_dt = df.at[idx, "Listing Date"]
            issue_price = df.at[idx, "Issue Price (₹)"]
            
            # Formulate date range (Listing date to Listing date + 15 days) to ensure we hit a week of trading
            dt_from = list_dt.strftime("%d-%m-%Y")
            dt_to = (list_dt + timedelta(days=15)).strftime("%d-%m-%Y")
            
            # API URL
            # Note: series argument is usually EQ or SM, sometimes NSE uses series=EQ for all historical cm
            url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22{series}%22]&from={dt_from}&to={dt_to}"
            
            if req_count > 0 and req_count % COOKIE_REFRESH == 0:
                print("  🔄 Refreshing cookies...")
                driver.get(NSE_HOME)
                time.sleep(3)
                
            print(f"[{req_count+1}] Fetching {symbol} (from {dt_from})...")
            driver.get(url)
            time.sleep(REQUEST_DELAY)
            req_count += 1
            
            try:
                body = driver.find_element(By.TAG_NAME, "body").text
                if not body or body.startswith("<"):
                    print(f"    ⚠ Non-JSON response for {symbol}")
                    # Try getting cookies again just in case
                    driver.get(NSE_HOME)
                    time.sleep(3)
                    continue
                    
                data = json.loads(body)
                records = data.get("data", [])
                
                # If series gave no results, try without series or with series=[%22EQ%22] for SME
                if not records and series == "SM":
                    alt_url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22EQ%22]&from={dt_from}&to={dt_to}"
                    driver.get(alt_url)
                    time.sleep(REQUEST_DELAY)
                    req_count += 1
                    body = driver.find_element(By.TAG_NAME, "body").text
                    try:
                        records = json.loads(body).get("data", [])
                    except Exception:
                        pass
                
                if records:
                    # Sort records by date ascending just to be sure (NSE usually returns latest first)
                    # NSE format: CH_TIMESTAMP: "13-Nov-2024"
                    records.sort(key=lambda x: pd.to_datetime(x.get("CH_TIMESTAMP"), errors='coerce'))
                    
                    # Day 1 close (first record usually matches listing date)
                    day1_close = records[0].get("CH_CLOSING_PRICE")
                    
                    # Week 1 close (5th trading day, index 4, or last if less than 5)
                    week_idx = min(4, len(records) - 1)
                    week_close = records[week_idx].get("CH_CLOSING_PRICE")
                    
                    if day1_close is not None:
                        day1_close = float(day1_close)
                        df.at[idx, "Listing Day Close (₹)"] = day1_close
                        df.at[idx, "Listing Day Gain (%)"] = ((day1_close - issue_price) / issue_price) * 100
                        success += 1
                        
                    if week_close is not None:
                        week_close = float(week_close)
                        df.at[idx, "Listing Week Close (₹)"] = week_close
                        df.at[idx, "Listing Week Gain (%)"] = ((week_close - issue_price) / issue_price) * 100
                        
                    gain_str = f"Day1={df.at[idx, 'Listing Day Gain (%)']:.1f}%" if pd.notna(df.at[idx, 'Listing Day Gain (%)']) else "N/A"
                    print(f"    ✅ {gain_str}")
                else:
                    print(f"    ⚠ No historical data found for {symbol}")
                    
                # Save aggressively
                if success > 0 and success % 20 == 0:
                    df.to_csv(OUTPUT_CSV, index=False)
                    
            except Exception as e:
                print(f"    ❌ Error parsing {symbol}: {e}")
                
        # Final save
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 Done! Fetched {success} successfully. Saved to {OUTPUT_CSV}")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
