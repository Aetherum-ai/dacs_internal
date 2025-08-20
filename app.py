from flask import Flask, render_template, request
from utils import eval_credit_score, eval_aum, build_portfolio
from utils import calculate_volatility_score_portfolio, eval_aqm
from utils import get_user_holdings, eval_lanw
from utils import plot_price
import pickle
import pandas as pd
import os


app = Flask(__name__)


# Load index data

btc_df = pd.read_csv('btc_data.csv', parse_dates=True, index_col='Date')
marketcap_df = pd.read_csv('total_marketcap_data.csv', parse_dates=True, index_col='Date')

index_dict = {
    "Bitcoin": btc_df,
    "Total Market Cap": marketcap_df
}

@app.route('/', methods=['GET', 'POST'])
def index():
    default_weights = {
        "credit": 25,
        "aum": 25,
        "vol_score": 20,
        "aqm": 15,
        "lanw": 15
        }

    weights = default_weights.copy()
    final_score = None
    selected_weights = []
    score_credit = None
    score_aum = None  
    score_vol_score = None
    score_aqm_score = None
    score_lanw = None
    portfolio_trend = None


    with open("coin_data.pkl", "rb") as f:
        coin_dict = pickle.load(f)

    coins = [
    "USDC", "BTC", "ETH", "ADA", "SOL", "DOT", "LINK", "AVAX", "DOGE", "SHIB",
    "UNI", "AAVE", "COMP", "MANA", "SAND", "ENJ", "ALGO", "XTZ", "BNB", "XRP",
    "ATOM", "ARB", "NEAR", "HBAR"
    ]

    if request.method == 'POST':
        # Sliders and inputs
        credit_score = int(request.form['credit_score']) 
        aum_raw = request.form['aum']
        age = int(request.form['age'])
        total_invested_raw = request.form['total_invested']

        aum = float(aum_raw.replace(",", ""))
        total_invested = float(total_invested_raw.replace(",", "")) 

        # Coins
        for coin in coins:
            if coin in request.form.getlist('coins'):
                value = request.form.get(f'value_{coin}')
                try:
                    selected_weights.append((float(value) / 100 ) if value else 0.0)
                except ValueError:
                    selected_weights.append(0.0)
            else:
                selected_weights.append(0.0)


        portfolio_df = build_portfolio(coin_dict, selected_weights, total_invested)

  


        # ---------- Score Weights and calculations ------------
            
        # ---------- Score Weights ------------

        # DACS weight inputs
        DACS_components = [
            "credit", "aum", "vol_score", "aqm", "lanw"
        ]
        

        for comp in DACS_components:
            field_name = f"weight_{comp}"
            raw = request.form.get(field_name)

            try:
                weights[comp] = float(raw) if raw else default_weights[comp]
            except ValueError:
                weights[comp] = default_weights[comp]
        
        # ---------- Score Calculations ------------


        # Credit Score and AUM
        score_credit = eval_credit_score(credit_score, weights['credit']) 
        score_aum = eval_aum(aum, age, weights['aum']) 

        # Volatility Score:
        total_data = pd.read_csv('total_marketcap_data.csv', index_col=0, parse_dates=True)
        score_vol_score = calculate_volatility_score_portfolio(portfolio_df, total_data, weights['vol_score']) 
        
        # AQM Score:
        aqm_data_df = pd.read_csv('aqm_scores.csv', index_col=0)
        aqm_data = aqm_data_df['aqm_score'].to_dict()
        score_aqm_score = eval_aqm(aqm_data, selected_weights, weights['aqm'])

        # LANW Score:
        traditional_assets = [
        "Cash", "Accounts_Receivables", "Blue_Chip_Equities", 
        "Mid_Cap_Equities", "Real_Estate", "Bonds", "Commodities"
        ]
        other_assets = {}
        for asset in traditional_assets:
            checkbox_name = f"other_{asset}"
            amount_name = f"amount_{asset}"

            if checkbox_name in request.form:
                try:
                    amount = float(request.form.get(amount_name, 0))
                    if amount > 0:
                        other_assets[asset.replace("_", " ")] = amount
                except ValueError:
                    continue  # skip invalid inputs


        holdings_dict = get_user_holdings(selected_weights, coin_dict, total_invested)

        score_lanw = eval_lanw(holdings_dict, other_assets, weights['lanw'], aum)




        # Final DACS Score:
        final_score = score_credit + score_aum + score_vol_score + score_aqm_score + score_lanw

        portfolio_viz = portfolio_df.copy()
        start_date = portfolio_viz.index.min()
        end_date = portfolio_viz.index.max()

        index_dict_1yr = {}
        for name, df in index_dict.items():
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = df.loc[start_date:end_date]
            index_dict_1yr[name] = df


        portfolio_trend = plot_price(portfolio_viz, index_dict_1yr)
    return render_template(
        'index.html', 
        score_credit = score_credit, 
        score_aum = score_aum, 
        score_vol_score = score_vol_score,
        score_aqm_score = score_aqm_score,
        score_lanw = score_lanw,
        final_score = final_score,
        coins = coins,
        weights = weights,
        portfolio_trend = portfolio_trend
        )

@app.route('/credit')
def credit_score():
    return render_template('credit_score.html')

@app.route('/vol', methods=['GET', 'POST'])
def vol_score():
    with open("coin_data.pkl", "rb") as f:
        coin_dict = pickle.load(f)

    coins = list(coin_dict.keys())
    selected_coin = None
    price_img = None
    start_date = None
    end_date = None


    if request.method == 'POST':
        selected_coin = request.form.get('coin')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        coin_df = coin_dict.get(selected_coin)

        # Slice coin data
        # Slice coin data
        try:
            if coin_df is not None:
                coin_df.index = pd.to_datetime(coin_df.index)

                # Create a local copy of the global index_dict
                local_index_dict = {
                    key: df.copy() for key, df in index_dict.items()
                }

                # If user selects a date range
                if start_date and end_date:
                    coin_df = coin_df.loc[start_date:end_date]
                    for key in local_index_dict:
                        local_index_dict[key].index = pd.to_datetime(local_index_dict[key].index)
                        local_index_dict[key] = local_index_dict[key].loc[start_date:end_date]
                else:
                    # Auto-slice based on coin_df range
                    coin_start = coin_df.index.min()
                    coin_end = coin_df.index.max()
                    for key in local_index_dict:
                        local_index_dict[key].index = pd.to_datetime(local_index_dict[key].index)
                        local_index_dict[key] = local_index_dict[key].loc[coin_start:coin_end]

            price_img = plot_price(coin_df, index_dict)
        except Exception as e:
            print(f"Plotting failed: {e}")

    return render_template('vol_score.html',
                           coins=coins,
                           selected_coin=selected_coin,
                           price_img=price_img,
                           start_date=start_date,
                           end_date=end_date)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug = False)
