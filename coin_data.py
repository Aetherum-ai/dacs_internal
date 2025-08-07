from utils import fetch_price, fetch_marketcap_history
from utils import calculate_AQM_coin
import pickle
import os
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import csv

# Step 1: Generate all data
coin_dict = {
    "USDC": fetch_price("USDC", period="7y"),
    "BTC": fetch_price("BTC", period="16y"),
    "ETH": fetch_price("ETH", period="10y"),
    "ADA": fetch_price("ADA", period="8y"),
    "SOL": fetch_price("SOL", period="5y"),
    "DOT": fetch_price("DOT", period="5y"),
    "LINK": fetch_price("LINK", period="8y"),
    "AVAX": fetch_price("AVAX", period="5y"),
    "DOGE": fetch_price("DOGE", period="12y"),
    "SHIB": fetch_price("SHIB", period="5y"),
    "UNI": fetch_price("UNI7083", period="5y"),
    "AAVE": fetch_price("AAVE", period="8y"),
    "COMP": fetch_price("COMP5692", period="4y"),
    "MANA": fetch_price("MANA", period="8y"),
    "SAND": fetch_price("SAND", period="5y"),
    "ENJ": fetch_price("ENJ", period="8y"),
    "ALGO": fetch_price("ALGO", period="6y"),
    "XTZ": fetch_price("XTZ", period="7y"),
    "BNB": fetch_price("BNB", period="8y"),
    "XRP": fetch_price("XRP", period="13y"),
    "ATOM": fetch_price("ATOM", period="5y"),
    "ARB": fetch_price("ARB11841", period="2y"),
    "NEAR": fetch_price("NEAR", period="5y"),
    "HBAR": fetch_price("HBAR", period="6y")
}


# API KEY:
api_key = 'c1bde33f-30b5-4dc1-8661-3af9b8bfab4e'

# One year date range (adjust as needed)
start_date = (date.today() - relativedelta(days = 363)).strftime('%Y-%m-%d')
end_date = date.today().strftime('%Y-%m-%d')

total_data = fetch_marketcap_history(api_key, start_date, end_date)
btc_data = fetch_price("BTC", period="max")
total_data.to_csv("total_marketcap_data.csv", index=True)

# Calculate AQM for each coin
aqm_data = calculate_AQM_coin(coin_dict, btc_data)
with open("aqm_scores.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["coin", "aqm_score"]) 
    for coin, score in aqm_data.items():
        writer.writerow([coin, score])

with open("coin_data.pkl", "wb") as f:
    pickle.dump(coin_dict, f)


print("Coin data updated successfully.")
