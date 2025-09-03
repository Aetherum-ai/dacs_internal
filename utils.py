import numpy as np
import pandas as pd
import requests
from datetime import date
from dateutil.relativedelta import relativedelta
import yfinance as yf
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import math
import io, base64


#<!--Fetching data functions--!>

def fetch_price(coin_id, period = 'max'):
    """
    Fetch historical daily prices for a cryptocurrency and return or add to a DataFrame.

    Parameters:
    - coin_id: coin ticker in Yahoo Finance e.g. 'BTC-USD'
    - period: Period to retrieve data from.

    Returns:
    - A dataframe containing price data with columns 'Date','Close' and 'Returns'
    """
    coin = yf.Ticker(f"{coin_id}-USD") # 'USD': coin-id against US Dollar by default
    coin_data = coin.history(period=f"{period}")
    coin_data.index = pd.to_datetime(coin_data.index).tz_localize(None)
    coin_data['returns'] = coin_data['Close'].pct_change()
    coin_data = coin_data.dropna()
    coin_data = coin_data[['Close', 'returns']]
    coin_data = coin_data.iloc[1:]
    return coin_data

def simulate_investment(asset_df, investment_amount = 100000):
    """
    Simulate buying an asset at the start of the period and holding it.
    Parameters:
    - asset_df (DataFrame): DataFrame with historical asset prices, must have 'Date', 'Close' and 'Returns' column.
    - investment_amount (float): how much the user invests on the first day
        
    Returns:
    DataFrame with same columns but 'Close' column is modified to show investment value change over time with first row set to 'investment_amount'.
    """
    df = asset_df.copy()

    # Normalize to start at 1.0 and scale to the investment amount
    initial_cap = df['Close'].iloc[0]
    df['index'] = df['Close'] / initial_cap
    df['portfolio_value'] = df['index'] * investment_amount

    # Building a dataframe with 'Close' and 'returns' column like the other ones in this notebook
    df['Close'] = df['portfolio_value']
    # Remove timezone info for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df['returns'] = df['Close'].pct_change()
    df = df.dropna()
    df = df[['Close', 'returns']]
    df = df.iloc[1:]

    return df

def fetch_marketcap_history(api_key, start_date, end_date):
    """
    Fetch historical total1 data for volatility score.

    Parameters:
    - api_key: Aetherum CoinMarketCap API key.
    - start_date: Start date for the historical data in 'YYYY-MM-DD' format.
    - end_date: End date for the historical data in 'YYYY-MM-DD'

    Returns:
    - A dataframe containing Total1 market cap data with columns 'Date', 'Close', and 'returns'.
    """
    url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/historical"
    
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key,
    }

    params = {
        'time_start': start_date,   # format: 'YYYY-MM-DD'
        'time_end': end_date,       # format: 'YYYY-MM-DD'
        'interval': 'daily',
        'convert': 'USD'
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    # Extract and convert to DataFrame
    quotes = data['data']['quotes']
    df = pd.DataFrame([{
        'Date': q['timestamp'][:10],
        'Close': q['quote']['USD']['total_market_cap'],
    } for q in quotes])
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['returns'] = df['Close'].pct_change()
    df = df.dropna()
    df = df.iloc[1:]

    return df


# <!--Portfolio building function--!>

def build_portfolio(coin_dict, weights, investment_amount):
    """
    Build a portfolio assuming investment of `investment_amount` today.
    Returns historical portfolio value over the past 1 year.

    Parameters:
    - coin_dict: dict of {coin_name: DataFrame}, each DataFrame must contain 'Close' and a datetime index
    - weights: list or array of asset weights (should sum to 1)
    - investment_amount: USD invested on the most recent date

    Returns:
    - portfolio_df: DataFrame with 'Close' (total value) and 'returns', indexed by Date
    """

    weights = np.array(weights)
    coins = list(coin_dict.values())

    # Determine 1-year window
    latest_date = min(coin.index.max() for coin in coins)
    start_date = latest_date - pd.DateOffset(years=1)
    coins_1yr = [coin[(coin.index >= start_date) & (coin.index <= latest_date)] for coin in coins]

    # Get latest closing prices
    latest_prices = [coin.iloc[-1]['Close'] for coin in coins_1yr]
    allocation = weights * investment_amount
    token_amounts = allocation / np.array(latest_prices)

    # Reconstruct historical total portfolio value
    portfolio_series = pd.Series(0, index=coins_1yr[0].index)
    for coin_df, amount in zip(coins_1yr, token_amounts):
        portfolio_series += coin_df['Close'] * amount

    portfolio_df = pd.DataFrame(index=portfolio_series.index)
    portfolio_df['Close'] = portfolio_series
    portfolio_df['returns'] = portfolio_df['Close'].pct_change()
    portfolio_df = portfolio_df.dropna()

    return portfolio_df
    
# <!--Plotting functions--!>

def render_plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

def plot_price(portfolio_df, index_dict=None):
    """
    Plot closing prices of the portfolio against multiple indices (e.g., BTC, Total Market Cap).

    Parameters:
    - portfolio_df: DataFrame with 'Close' prices
    - index_dict: dict of name -> DataFrame, each with a 'Close' column
    """
    portfolio_df = simulate_investment(portfolio_df)

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df.index, portfolio_df['Close'], label='Asset Price', color='green')

    if index_dict:
        for name, df in index_dict.items():
            df = simulate_investment(df)
            plt.plot(df.index, df['Close'], label=f'{name} Price')

    plt.title('Price Trend: Portfolio vs Indices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return render_plot_to_base64()
# <!--Core DACS functions:--!>
    
def eval_credit_score(credit_score, weight):
    """
    Evaluate the user's credit score and return a score
    """
    perfect_credit = 850
    return (credit_score / perfect_credit) * weight 

average_savings_by_age = {
    'Less than 35': 49130,
    '35-44': 141520,
    '45-54': 313220,
    '55-64': 537560,
    '65-74': 609230,
    '75 or older': 462410
}

def eval_aum(total_aum, age, score):
    """
    Evaluate the user's total Assets Under Management (AUM) and return a score based on age
    """
    if age < 35:
        if total_aum < average_savings_by_age['Less than 35']:
            return 0.5 * score
        elif total_aum <= 1.2 * average_savings_by_age['Less than 35']:
            return 0.8 * score
        elif total_aum >= 1.2 * average_savings_by_age['Less than 35']:
            return score 
        
    elif age < 45:
        if total_aum < average_savings_by_age['35-44']:
            return 0.5 * score
        elif total_aum <= 1.2 * average_savings_by_age['35-44']:
            return 0.8 * score
        elif total_aum >= 1.2 * average_savings_by_age['35-44']:
            return score     
        
    elif age < 55:
        if total_aum < average_savings_by_age['45-54']:
            return 0.5 * score
        elif total_aum <= 1.2 * average_savings_by_age['45-54']:
            return 0.8 * score
        elif total_aum >= 1.2 * average_savings_by_age['45-54']:
            return score     
        
    elif age < 65:
        if total_aum < average_savings_by_age['55-64']:
            return 0.5 * score
        elif total_aum <= 1.2 * average_savings_by_age['55-64']:
            return 0.8 * score
        elif total_aum >= 1.2 * average_savings_by_age['55-64']:
            return score     

    elif age < 75:
        if total_aum < average_savings_by_age['65-74']:
            return 0.5 * score
        elif total_aum <= 1.2 * average_savings_by_age['65-74']:
            return 0.8 * score
        elif total_aum >= 1.2 * average_savings_by_age['65-74']:
            return score     
    else:
        if total_aum < average_savings_by_age['75 or older']:
            return 0.5 * score
        elif total_aum <= 1.2 * average_savings_by_age['75 or older']:
            return 0.8 * score
        elif total_aum >= 1.2 * average_savings_by_age['75 or older']:
            return score     


# <!--Volatility score functions--!>
def calculate_volatility_score(asset_data, index_data):
    """
    Calculates the *raw* volatility score for a single coin compared to an index.
    
    Inputs:
    - asset_data: DataFrame with datetime index, 'Close' and 'returns'
    - index_data: DataFrame with datetime index, 'Close' and 'returns'

    Returns:
    - raw volatility score (float)
    """

    # Ensure datetime index
    asset_data.index = pd.to_datetime(asset_data.index)
    index_data.index = pd.to_datetime(index_data.index)

    latest_date = min(asset_data.index.max(), index_data.index.max())
    one_year_ago = latest_date - pd.DateOffset(years=1)

    # Slice both dataframes to the last 1 year
    portfolio_recent = asset_data[(asset_data.index >= one_year_ago) & (asset_data.index <= latest_date)]
    index_recent = index_data[(index_data.index >= one_year_ago) & (index_data.index <= latest_date)]

    # Align by common dates
    common_dates = portfolio_recent.index.intersection(index_recent.index)
    portfolio_aligned = portfolio_recent.loc[common_dates]
    index_aligned = index_recent.loc[common_dates]

    if len(common_dates) < 2:
        raise ValueError("Not enough overlapping data in the past year to calculate volatility.")

    # Normalize 'Close' prices
    coin_norm = portfolio_aligned['Close'] / portfolio_aligned['Close'].iloc[0]
    index_norm = index_aligned['Close'] / index_aligned['Close'].iloc[0]

    # Relative Volatility
    coin_volatility = np.std(coin_norm)
    index_volatility = np.std(index_norm)
    relative_volatility = coin_volatility / index_volatility if index_volatility != 0 else 0

    # Beta Ratio
    index_var = np.var(index_aligned['returns'])
    covar = np.cov(portfolio_aligned['returns'], index_aligned['returns'])[0][1]
    beta_ratio = covar / index_var if index_var != 0 else 0

    # Mean normalized difference
    avg_norm_diff = np.mean(np.abs(coin_norm - index_norm))

    # Raw score 
    volatility_score = relative_volatility * (1 - np.abs(beta_ratio)) + avg_norm_diff

    return volatility_score


def eval_vol_score(vol_score, weight):
    """
    Evaluates a raw volatility score into a final bounded score based on thresholds.
    
    Inputs:
    - vol_score: the raw volatility score (float)
    - weight: Weight in the DACS


    Returns:
    - evaluated volatility score (float)
    """
    epsilon = 0.05
    if vol_score < epsilon:
        return weight  # idealized case
    if vol_score < 0.7:
        return weight  # best score achieved by very stable coins
    if 0.7 <= vol_score <= 1:
        return vol_score * weight  # good score, scale it
    return (1 / vol_score) * weight  # normalize higher volatility

# <!--Asset Quality Score--!>

def calc_r_squared(asset_df, index_df):
    """
    This function calculates the R-squared value between two DataFrames. It is created for 
    cleaner code and more abstraction because calculating R-squared involves fitting a linear regression


    Input: 
    asset_df: a portfolio/coin df that has 'Close' and 'returns' columns
    index_df: a market index df that has 'Close' and 'returns' columns

    Returns:
    R-squared value (float) between the two DataFrames.
    """
    common_dates = asset_df.index.intersection(index_df.index)
    y_series = asset_df.loc[common_dates]['returns'].dropna()
    x_series = index_df.loc[common_dates]['returns'].dropna()
    
    # Final alignment in case dropna created mismatch
    common_dates = y_series.index.intersection(x_series.index)
    y = y_series.loc[common_dates].tolist()
    x = x_series.loc[common_dates].tolist()

    if len(x) == 0 or len(y) == 0:
        return 0
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    cov_xy = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
    std_x = (sum((x[i] - x_mean) ** 2 for i in range(len(x)))) ** 0.5
    std_y = (sum((y[i] - y_mean) ** 2 for i in range(len(y)))) ** 0.5

    if std_x == 0 or std_y == 0:
        return 0

    correlation = cov_xy / (std_x * std_y)
    r_squared = correlation ** 2
    return r_squared

memecoins = {"DOGE", "SHIB", "FLOKI", "BONK", "WIF"}
def calculate_AQM_coin(coin_dict, index_data, total_data):
    """ 
    Calculates the Asset Quality Matrix (AQM) for each coin.

    Inputs:
    - coin_dict: dict mapping coin_name -> DataFrame with 'returns' and 'Close', indexed by Date
    - index_data: DataFrame for market index with 'returns' and 'Close', indexed by Date
    - weight: tuple or list of two floats (w1, w2) for Sharpe*R² and Sortino respectively
    - coin_creation_years: dict mapping coin_name -> creation year (e.g. {'BTC': 2009, 'WIF': 2024})

    Returns:
    - aqm_scores: dict mapping coin_name -> AQM score
    """

    BTC_YEAR = 2009
    aqm_scores = {}


    for name, coin_df in coin_dict.items():

        # Align dates
        common_dates = coin_df.index.intersection(index_data.index)
        coin_aligned = coin_df.loc[common_dates]
        index_aligned = index_data.loc[common_dates]

        # BTC is 0 in AQM calc but will be changed to 1 
        if name in {"BTC", "USDC", "DAI"}:
            aqm_scores[name] = 1.0
            continue

        if len(coin_aligned) < 2:
            aqm_scores[name] = 0
            continue

        # Expected returns
        port_returns = np.mean(coin_aligned['returns'])
        index_returns = np.mean(index_aligned['returns'])

        # Sharpe Ratio
        sd_portfolio = np.std(coin_aligned['returns'])
        sharpe_ratio = (port_returns - index_returns) / sd_portfolio if sd_portfolio != 0 else 0

        # Sortino Ratio
        downside_returns = coin_aligned['returns'][coin_aligned['returns'] < 0]
        sd_downside = np.std(downside_returns)
        sortino_ratio = (port_returns - index_returns) / sd_downside if sd_downside != 0 else 0

        # R² (Assume calc_r_squared is defined elsewhere)
        r_squared = calc_r_squared(coin_aligned, index_aligned)

        # Time penalty
        coin_year = int(coin_df.index.min().year)
        t = max(coin_year - BTC_YEAR, 1)
        t = np.exp(-0.5 * t) # Penalizes newer coins by exponential decay

        # AQM formula
        aqm =  ((sharpe_ratio * r_squared) +  sortino_ratio)
        aqm = t * aqm
        aqm = aqm * 100 # Scale for interpretability
        
        if name in memecoins:
            vol_score = calculate_volatility_score(coin_aligned, total_data)
            if vol_score > 0 :
                aqm *= (1 / vol_score)

        aqm_scores[name] = aqm


    return aqm_scores

def eval_aqm(aqm_data, selected_weights, aqm_weight):
    """
    Evaluates overall portfolio AQM which is simply a weighted sum of the user's value per asset and the
    asset's AQM score.

    Inputs:
    - aqm_data: dict mapping coin_name -> AQM score for each respective coin
    - selected_weights: the amount in percentage the user owns of a specific coin
    - aqm_weight: AQM weight for the DACS

    returns:
    - AQM score for the portfolio (float)
    """
     # Ensure weights and AQM data align
    coin_names = list(aqm_data.keys())
    if len(selected_weights) != len(coin_names):
        print(f"Length of selected_weights: {len(selected_weights)}")
        print(f"Length of aqm_data: {len(aqm_data)}")


    # Compute weighted average AQM
    overall_aqm = 0.0

    for weight, coin in zip(selected_weights, coin_names):
        aqm = aqm_data.get(coin, 0.0)
        overall_aqm += weight * aqm


    # Apply final AQM weight
    return overall_aqm * aqm_weight



def get_user_holdings(selected_weights, coin_dict, total_invested):
    """
    Returns a dictionary of the user's actual dollar holdings in each coin based on
    selected weights and total investment amount.

    Parameters:
    - selected_weights: list of floats representing portfolio weights for each coin (same order as coin_dict)
    - coin_dict: dict mapping coin_name -> DataFrame (only keys are used for coin order)
    - total_invested: float, total amount user is investing

    Returns:
    - holdings_dict: dict mapping coin_name -> dollar amount owned (only for coins with nonzero ownership)
    """

    coin_names = list(coin_dict.keys())
    if len(selected_weights) != len(coin_names):
        raise ValueError("selected_weights must match number of coins in coin_dict")

    holdings_dict = {}
    for coin, weight in zip(coin_names, selected_weights):
        amount = weight * total_invested
        if amount > 0:
            holdings_dict[coin] = amount

    return holdings_dict

# Liquidity-Adjusted Net worth:

# Liquidity scores per coin:
coin_liquidity_classification = {
    1.0:["USDC"],
    0.75: [
        "BTC", "ETH", "BNB", "XRP", "SOL", "USDC", "ADA", "LINK", "AVAX", "DAI", "LTC"
    ],
    0.65: [
        "MATIC", "UNI", "ATOM", "ARB", "NEAR", "HBAR", "ALGO", "XTZ", "AAVE", "SUI", "UNI", "TAO", "FET", "INJ"
    ],
    0.35: [
         "MANA", "SAND", "ENJ", "DOGE", "SHIB", "FLOKI", "HYPE", "BONK", "WIF"
    ]
}

# Other Liquidty scores per asset class:
other_asset_classifications = {
    1.0: [
        "Cash", "Accounts Receivables" 
    ],
    0.85: [
        "Blue Chip Equities"
    ], 
    0.80: [
        "Mid Cap Equities"
    ],
    0.30: [
        "Real Estate", "Bonds", "Commodities"
    ]
}

def eval_lanw(holdings_dict, other_assets, weight_lanw, total_aum):
    """
    Calculates the Liquidity Adjusted Net worth of the user
    Inputs:
    - selected_weights: a list storing the amount in percentage the user owns of a specific coin
    - coin_dict: dict mapping coin_name -> DataFrame with 'returns' and 'Close', indexed by Date
    - other_assets: another list storing the amount in percentage the user owns of other assets
    - total_aum: total assets under management

    Returns:
    - lanw_score: Liquidity Adjusted Net Worth score (float)
    """
    total_lanw = 0.0
    
    # Crypto portion
    for coin, amount in holdings_dict.items():
        for liquidity_score, coin_list in coin_liquidity_classification.items():
            if coin in coin_list:
                total_lanw += liquidity_score * amount
                break

    # Traditional asset portion
    for asset, amount in other_assets.items():
        for liquidity_score, asset_list in other_asset_classifications.items():
            if asset in asset_list:
                total_lanw += liquidity_score * amount
                break
    
    if total_lanw > total_aum:
        raise("Total LANW cannot exceed total AUM")
    lanw_score = (total_lanw / total_aum)
    
    return lanw_score * weight_lanw

# Volatility Premium:

def eval_volatility_premium(vol_score, vol_score_weight):
    """
    Calculates Volatility Premium 
    Inputs:
    - Volatility Score
    - Volatility Score weight

    Returns:
    - Volatility Premium (%)
    """
    vol_prem = 0
    if vol_score == 1 * vol_score_weight :
        vol_prem = 0
    elif (vol_score < 1 * vol_score_weight) and (vol_score >= 0.75 * vol_score_weight) :
        vol_prem = 0.25
    elif (vol_score < 0.75 * vol_score_weight ) and (vol_score >= 0.5 * vol_score_weight) :
        vol_prem = 0.5
    elif (vol_score < 0.5 * vol_score_weight):
        vol_prem = 1
    return vol_prem