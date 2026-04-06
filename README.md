# 📈 NSE IPO Analyzer & Predictor (2016-2026)

A comprehensive data engineering and machine learning project to scrape, analyze, and predict the Listing Day profitability of Indian IPOs (Mainboard and SME) from the National Stock Exchange (NSE).

## 🚀 Project Overview

This repository captures a fully automated pipeline for IPO investment analysis:
1. **Data Extraction**: Custom scrapers fetching structured historical IPO data and issue insights.
2. **Exploratory Data Analysis (EDA)**: Visualizing and uncovering trends, hit rates, seasonality, and correlations over a 10-year span.
3. **Machine Learning Classifier**: Building traditional statistical models (LogReg, Random Forest, GBM) to predict whether an upcoming IPO will list at a premium.

---

## 📂 Repository Structure

- `nse_ipo_scraper.py` / `chittorgarh_scraper.py`: Scripts used to fetch, parse, and scrape historical IPO issues, dates, pricing bounds, and registrar data.
- `fetch_listing_gains_browser.py`: Fetches real-world listing and week-closing prices.
- `clean_and_merge.py`: Cleans and joins multiple disparate datasets into our main `nse_ipo_merged.csv`.
- `data_analysis.py`: Generates the **16 high-quality analysis charts** and visual summaries found in `analysis_output/`.
- `ipo_classifier.py`: Trains mathematical models on our features to predict profitability outcomes.
- `requirements.txt`: Strict pip-freeze list for exact environment replication.

---

## 📊 Key Findings from 1,200+ IPOs 

*Out of 1,207 total IPOs across EQ and SME boards, 794 had full clean listing data spanning 2016-2026.*

* **2021-2024 Boom:** The market completely shifted post-pandemic. The overall "Hit Rate" (percentage of IPOs providing a positive return on listing day) went from **32% in 2019** to **84% in 2024**.
* **SMEs vs Mainboards:** SME IPOs offer an average of +28.5% gains compared to the Mainboard's +9.8%. However, they come with substantial risk tail-ends (some losing over 95% value).
* **Correlation Reality Check:** Surprisingly, total over-subscription size has a very weak correlation ($r \approx 0.10$) with actual listing gains. Highly subscribed does *not* automatically mean highly profitable.
* **Sell on Listing Day:** Listing Day gains and Week 1 closing gains are fundamentally identical ($r = 0.96$). Almost all price discovery happens immediately upon listing.

---

## 🧠 Machine Learning: Profitability Predictor

Our goal was to build a strict mathematical classification algorithm (0 for flat/loss, 1 for profitable listing) using only pre-listing knowledge like Issue Price, Subscriptions, Issue Size, and Timestamps.

**Model Selection:** We evaluated Logistic Regression, Gradient Boosting, and **Random Forests**. 
> While Logistic Regression had higher naked accuracy (85.2%), it wildly overpredicted wins. The **Random Forest** classifier with Class Balancing (83.9% Accuracy, `0.831` ROC-AUC) successfully identified **66% of the unprofitable IPOs** while still correctly tagging 88% of the winners.

### Top Predictive Features (Random Forest)
1. **Issue Price (₹) (~17%)**: Directly correlates with whether an IPO is Mainboard vs SME.
2. **Launch Year (~16%)**: Macro market sentiment completely dominates baseline probabilities.
3. **Issue Size (~9.5%)**: Overall liquidity buffer.
4. **Retail Subscription (~9%)**: Outweighed QIB (Institutional) subscription as an indicator.

---

## ⚙️ How to Replicate

**1. Clone and Setup Environment**
```bash
conda create -n IPOA python=3.14
conda activate IPOA
pip install -r requirements.txt
```

**2. Run the Analysis Module**
```bash
python3 data_analysis.py
```
*(This rebuilds all 16 charts and JSON summaries into the `analysis_output/` folder based on the CSV data).*

**3. Run the ML Classifier**
```bash
python3 ipo_classifier.py
```
*(This will trigger processing of the dataset, target split creation, and output precision/recall models alongside ROC-AUC scores).*


## Project by: 
1. 23DCS004 Ankush Thakur.
2. 23DCS006 Arnav Sharma.
3.  23DCS007 Divye Vaibhav Mishra.
4. 23DCS014 Kanishak.
5. 23DCS021 Raghav Sharma.
6. 23DCS024 Sarthak Katiyar.
