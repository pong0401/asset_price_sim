import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import os
import time
from datetime import datetime, timezone, timedelta
import yfinance as yf
from scipy import optimize
import plotly.graph_objects as go
import pytz
import requests
from config import *

# Directory to save crypto data files
data_dir = "crypto_data"

# Ensure the directory exists
os.makedirs(data_dir, exist_ok=True)


# # List of crypto symbols
# crypto_symbols = [
#     'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD',
#     'SOL-USD', 'ADA-USD', 'DOT-USD', 'LTC-USD', 'SHIB-USD',
#     'TRX-USD', 'AVAX-USD', 'UNI-USD', 'LINK-USD', 'ATOM-USD',
#     'ETC-USD', 'XLM-USD', 'BCH-USD', 'APT-USD', 'FIL-USD',
#     'ALGO-USD', 'VET-USD', 'ICP-USD', 'NEAR-USD', 'QNT-USD',
#     'FTM-USD', 'EOS-USD', 'SAND-USD', 'AAVE-USD', 'MANA-USD',
#     'THETA-USD', 'XTZ-USD', 'EGLD-USD', 'AXS-USD', 'CHZ-USD',
#     'RPL-USD', 'CAKE-USD', 'KAVA-USD', 'ZIL-USD',
#     'XEC-USD', 'BAT-USD', 'CRV-USD', 'DYDX-USD', 'GALA-USD',
#     'STX-USD', 'BAL-USD', 'BONK-USD',
#     'FLOKI-USD', 'LDO-USD', 'KAS-USD',
#     'INJ-USD', 'ARB-USD', 'OP-USD', 'WLD-USD',
#      'LRC-USD', 'ENS-USD', 'FXS-USD', 'MINA-USD',
#     'OSMO-USD', 'ROSE-USD', 'CELO-USD', '1INCH-USD', 'GNO-USD',
#     'KNC-USD', 'ANKR-USD', 'COTI-USD', 'SXP-USD', 'YFI-USD',
#      'SNX-USD', 'UMA-USD', 'ZRX-USD', 'BNT-USD',
#     'REN-USD', 'CTSI-USD', 'DGB-USD', 'STORJ-USD', 'CVC-USD',
#     'NKN-USD', 'SC-USD', 'ZEN-USD', 'KMD-USD', 'ARK-USD',
#     'DENT-USD', 'FUN-USD', 'MTL-USD', 'STMX-USD', 'POWR-USD',
#     'REQ-USD', 'BAND-USD', 'RLC-USD', 'MLN-USD', 'ANT-USD',
#     'NMR-USD', 'LPT-USD', 'OXT-USD', 'DIA-USD', 'TRB-USD',
#     'WNXM-USD', 'AVA-USD', 'LIT-USD', 'PHA-USD',
#     'AKRO-USD', 'RIF-USD', 'DOCK-USD', 'DODO-USD',
#     'ALPHA-USD', 'BEL-USD', 'TWT-USD', 'FOR-USD', 'FRONT-USD',
#     'UNFI-USD', 'FLM-USD', 'SUSHI-USD', 'YFII-USD',
#     'CREAM-USD', 'KP3R-USD', 'CVP-USD', 'SUN-USD'
# ]

def current_USDTHB():
    url="https://api.exchangerate-api.com/v4/latest/USD"
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    data = response.json()
    return data['rates']['THB']

# Variance Gamma Functions
def log_return(price_df):
    price_series = price_df['Close']
    log_returns = np.log(price_series / price_series.shift(1)).values.ravel()  # Flatten to 1D
    return pd.DataFrame(log_returns, index=price_df.index)

def fit_moments(x):
    mu = np.mean(x)
    sigma_squared = np.mean((x - mu) ** 2)
    beta = np.mean((x - mu) ** 3) / np.mean((x - mu) ** 2) ** 1.5
    kapa = np.mean((x - mu) ** 4) / np.mean((x - mu) ** 2) ** 2
    sigma = sigma_squared**0.5
    nu = kapa / 3.0 - 1.0
    theta = sigma * beta / (3.0 * nu)
    c = mu - theta
    return (c, sigma, theta, nu)

def neg_log_likelihood(data, par):
    if (par[1] > 0) & (par[3] > 0):
        return -np.sum(np.log([1e-10 + abs(data).mean()]))  # Simplified
    else:
        return np.inf

def fit_ml(data, maxiter=1000):
    par_init = np.array(fit_moments(data))
    par = optimize.fmin(lambda x: neg_log_likelihood(data, x), par_init, maxiter=maxiter, disp=False)
    return tuple(par)

def gen_vargamma_pct_change(n, c, sigma, theta, nu, time_to_maturity):
    mu = c + theta
    shape = time_to_maturity / nu
    scale = nu
    gamma_rand = np.random.gamma(shape=shape, scale=scale, size=n)
    dW = np.random.normal(size=n)
    pct_changes = sigma * np.sqrt(gamma_rand) * dW
    return pct_changes

def simulate_vg_scenarios_pct(S0, c_fit, sigma_fit, theta_fit, nu_fit, steps, num_scenarios, start_date):
    dates = [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i) for i in range(steps)]
    pct_changes = np.zeros((steps, num_scenarios))
    for i in range(steps):
        pct_change = gen_vargamma_pct_change(num_scenarios, c_fit, sigma_fit, theta_fit, nu_fit, time_to_maturity=1)
        pct_changes[i, :] = pct_change
    pct_changes_df = pd.DataFrame(pct_changes, index=dates, columns=[f'Scenario_{i+1}' for i in range(num_scenarios)])
    return pct_changes_df
# Function to fetch data and save to a file

# def is_file_updated_recently(filename):
#     """
#     Check if the file contains data up to the last hour.
#     """
#     if not os.path.exists(filename):
#         return False
    
#     # Load the file and parse timestamps
#     df = pd.read_csv(filename)
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
    
#     # Ensure timestamps are timezone-aware (assume UTC for file data)
#     if df['timestamp'].dt.tz is None:
#         df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
#     # Get the most recent timestamp in the file
#     last_timestamp = df['timestamp'].max()
    
#     # Calculate the time threshold (one hour ago, in UTC)
#     utc = pytz.UTC
#     one_hour_ago = datetime.now(utc) - timedelta(hours=1)
#     print(filename,last_timestamp,one_hour_ago,last_timestamp >= one_hour_ago)
#     # Check if the most recent timestamp is within the last hour
#     return last_timestamp >= one_hour_ago

# # Function to fetch and save data
# def fetch_and_save_data(symbols, limit=30):
#     """
#     Fetch historical data for multiple symbols, save to files, and return as a dictionary of DataFrames.
#     """
#     end_date = datetime.now()
#     start_date = end_date - timedelta(days=limit)

#     # Fetch data
#     data = yf.download(
#         tickers=symbols,
#         start=start_date.strftime('%Y-%m-%d'),
#         end=end_date.strftime('%Y-%m-%d'),
#         interval='1h',
#         group_by='ticker',
#         auto_adjust=True,
#         threads=True
#     )

#     crypto_data = {}

#     # Handle the multi-index columns returned by yfinance
#     if isinstance(data.columns, pd.MultiIndex):
#         for symbol in symbols:
#             if symbol in data.columns.levels[0]:  # Check if symbol has data
#                 symbol_data = data[symbol].dropna().reset_index()
#                 symbol_data.columns = ['timestamp'] + list(symbol_data.columns[1:])
#                 crypto_data[symbol] = symbol_data
#                 filename = f"{data_dir}/{symbol.replace('-', '_')}.csv"
#                 symbol_data.to_csv(filename, index=False)
#     else:
#         # If data does not have MultiIndex, handle it as a single DataFrame
#         for symbol in symbols:
#             if not data.empty and symbol in data:
#                 symbol_data = data[[col for col in data.columns if col.startswith(symbol)]].dropna().reset_index()
#                 symbol_data.columns = ['timestamp'] + list(symbol_data.columns[1:])
#                 crypto_data[symbol] = symbol_data
#                 filename = f"{data_dir}/{symbol.replace('-', '_')}.csv"
#                 symbol_data.to_csv(filename, index=False)
    
#     return crypto_data

# # Load or fetch data
# def load_or_fetch_data(symbols):
#     """
#     Load data from existing files or fetch new data for outdated files.
#     """
#     crypto_data = {}
#     outdated_symbols = []

#     # Check files
#     for symbol in symbols:
#         filename = os.path.join(data_dir, f"{symbol.replace('-', '_')}.csv")
#         if is_file_updated_recently(filename):
#             crypto_data[symbol] = pd.read_csv(filename)  # Load current file
#         else:
#             outdated_symbols.append(symbol)  # Mark as outdated

#     # Fetch data for outdated symbols
#     if outdated_symbols:
#         print(f"Fetching data for outdated symbols: {', '.join(outdated_symbols)}")
#         fetched_data = fetch_and_save_data(outdated_symbols)
#         for symbol, data in fetched_data.items():
#             crypto_data[symbol] = data

#     return crypto_data

def update_crypto_data(file_path, ticker):
    """
    Update the cryptocurrency data file with the latest data and return the updated DataFrame.
    
    Parameters:
        file_path (str): Path to the CSV file.
        ticker (str): Ticker symbol for the cryptocurrency (e.g., "BTC-USD").
        
    Returns:
        DataFrame: Updated DataFrame with the latest data.
    """
    # Load existing data
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
    else:
        data = pd.DataFrame()

    # Get current time with timezone
    current_time = datetime.now(pytz.utc)
    current_hour = current_time.replace(minute=0, second=0, microsecond=0)
    #start_date = current_time - timedelta(days=1)
    if not data.empty:
        last_date = data.index.max()

        # Ensure last_date is timezone-aware (convert to UTC)
        if last_date.tzinfo is None:
            last_date = last_date.tz_localize("UTC")
        #print("last_date",last_date,"curent hour",current_hour)
        # Check if last_date is more than 1 hour behind current_hour
        if (current_time - last_date).total_seconds() > 3600:
            end_period = '1d'
        else:
            #print(f"Data for {ticker} is already up-to-date.")
            return data  # No update needed
    else:
        # Default to fetching the last 30 days if the file is empty
        #start_date = current_time - timedelta(days=30)
        end_period='1mo'

    # Convert start_date and current_time to 'YYYY-MM-DD' format
    #start_date = start_date.strftime("%Y-%m-%d")
    
    #print(start_date,end_date)
    # Fetch updated data
    updated_data = yf.Ticker(ticker).history(period=end_period, interval="1h")
    if updated_data.empty:
        print(f"No new data found for {ticker}.")
        return data
    # print("data----------")
    # print(data)
    # print("update data---")
    # print(updated_data)

    updated_data = updated_data.rename_axis("timestamp").reset_index()
    updated_data = updated_data[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    updated_data["timestamp"] = pd.to_datetime(updated_data["timestamp"])
    updated_data.set_index("timestamp", inplace=True)

    # Combine old and new data
    combined_data = pd.concat([data, updated_data]).drop_duplicates().sort_index()

    # Save updated data back to the file
    combined_data.to_csv(file_path)
    print(f"Updated data saved for {ticker}.")

    return combined_data


def fetch_update_data():
    crypto_data={}
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            ticker = file_name.replace("_", "-").replace(".csv", "")  # Convert file name to ticker
            file_path = os.path.join(data_dir, file_name)
            #print(f"Processing {file_name}...")
            updated_df = update_crypto_data(file_path, ticker)
            crypto_data[ticker] = updated_df  # Add updated DataFrame to the dictionary
    return crypto_data

# Function to filter data for the last 7 days
def filter_last_7_days(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df[df['timestamp'] >= (datetime.now(timezone.utc) - pd.Timedelta(days=7))]

def find_last_trigger_date_and_price(data, x_days, volume_increase_pct, holding_period):
    # Ensure the index is a DatetimeIndex
    if data.index.dtype != 'datetime64[ns]':
        data.index = pd.to_datetime(data.index)
    # if not isinstance(data.index, pd.DatetimeIndex):
    #     data.set_index('timestamp', inplace=True)

    # Check if the index is timezone-naive
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')  # Localize to UTC if naive

    # Convert to the desired timezone (GMT+7)
    data.index = data.index.tz_convert('Asia/Bangkok')

    # Calculate X-day high (excluding the current day)
    data['x_day_high'] = data['Close'].shift(1).rolling(window=x_days).max()

    # Calculate volume percentage increase
    data['volume_pct_increase'] = (data['Volume'] - data['Volume'].shift(1)) / data['Volume'].shift(1) * 100

    # Trigger condition: Price above X-day high and volume increased by X%
    data['trigger'] = (data['Close'] > data['x_day_high']) & (data['volume_pct_increase'] > volume_increase_pct)

    # Find the last trigger date and price
    if data['trigger'].any():
        last_trigger_row = data[data['trigger']].iloc[-1]
        last_trigger_date = last_trigger_row.name
        last_trigger_price = last_trigger_row['Close']
    else:
        last_trigger_date = None
        last_trigger_price = None

    return last_trigger_date, last_trigger_price



# Streamlit App Layout
st.title("Crypto Strategy Performance and Price Simulation")
st.text("Analyze crypto strategies and simulate future price scenarios.")
st.text("Backtest strategy ด้วย ราคาทำ new high ในช่วง X วัน และ Volume ขึ้น Y% ข้อมูล 1 ปีที่ผ่านมา และ จำนวนวันที่ Hold order z ชั่วโมง")
st.text("และทำการ Monitor ราคา และ Volume ถ้าเข้าเงื่อนไขจากการ Backtest จะนำมา List ในตารางด้านล่าง โดยบอกวันที่มีสัญญาณ Trigger")
st.text("ช่วง Bull run ตอนที่ราคา BTC ATH ,เหรียญอื่นจะขึ้นตาม ช่วงที่หมด Bull run ไม่ควรใช้ strategy นี้ และใช้เพื่อการศึกษาเท่านั้น")
# Section 1: Strategy Analysis (Accuracy and Return Tables)
st.subheader("Crypto Strategy Signal Trigger for Short Term Trader")
param_result_file = "param_result_with_TP_SL.csv"
usd_to_thb_rate=current_USDTHB()
if os.path.exists(param_result_file):
    crypto_data = fetch_update_data()
    comparison_df=pd.read_csv(param_result_file,index_col=0)
    if crypto_data is not None:
        accuracy_results = []
        #return_results = []
        best_df=comparison_df.copy()
        # Iterate through symbols and calculate results
        for symbol, df in crypto_data.items():
            # Best accuracy parameters
            best_accuracy_row = best_df[best_df['Symbol'] == symbol]
            if best_accuracy_row.empty:
                print(f"No matching row found for {symbol} in best_df.")
                continue

            # Extract parameters safely
            best_accuracy_params = {
                'x_hours': best_accuracy_row['High_in_x_hours'].values[0],
                'volume_increase_pct': best_accuracy_row['Volume_Increase_Pct'].values[0],
                'holding_period': best_accuracy_row['Holding_Period'].values[0],
                'TP(%)': best_accuracy_row['TP(%)'].values[0],
                'SL(%)': best_accuracy_row['SL(%)'].values[0],
                'Num_Signals': best_accuracy_row['Num_Signals'].values[0],
                'Total_Return_No_TP_SL': best_accuracy_row['AVG_Return_No_TP_SL'].values[0],
                'Accuracy_No_TP_SL': best_accuracy_row['Accuracy_No_TP_SL'].values[0],
                'Total_Return_With_TP_SL': best_accuracy_row['AVG_Return_With_TP_SL'].values[0],
                'Accuracy_With_TP_SL': best_accuracy_row['Accuracy_With_TP_SL'].values[0],
            }

            # Ensure Num_Signals is valid
            num_signals = best_accuracy_params['Num_Signals']
            if num_signals == 0 or pd.isna(num_signals):
                avg_total_return_no_tp_sl = None
                avg_total_return_with_tp_sl = None
            else:
                avg_total_return_no_tp_sl = (
                    best_accuracy_params['Total_Return_No_TP_SL'] / num_signals
                )
                avg_total_return_with_tp_sl = (
                    best_accuracy_params['Total_Return_With_TP_SL'] / num_signals
                )

            # Find last trigger date and price
            last_trigger_date, last_trigger_price = find_last_trigger_date_and_price(
                df.copy(),
                best_accuracy_params['x_hours'],
                best_accuracy_params['volume_increase_pct'],
                best_accuracy_params['holding_period'],
            )

            # Append results
            accuracy_results.append({
                'Symbol': symbol,
                'Last_Trigger_Date': last_trigger_date,
                'Price(THB)': last_trigger_price*usd_to_thb_rate,
                'High_in_x_hours': best_accuracy_params['x_hours'],
                'Volume_Increase_Pct': best_accuracy_params['volume_increase_pct'],
                'Holding_hours': best_accuracy_params['holding_period'],
                'TP(%)': best_accuracy_params['TP(%)'],
                'SL(%)': best_accuracy_params['SL(%)'],
                'Num_Signals': num_signals,
                'AVG_Return_No_TP_SL': avg_total_return_no_tp_sl,
                'Accuracy_No_TP_SL': best_accuracy_params['Accuracy_No_TP_SL'],
                'AVG_Return_With_TP_SL': avg_total_return_with_tp_sl,
                'Accuracy_With_TP_SL': best_accuracy_params['Accuracy_With_TP_SL'],
            })

        # Create DataFrame
        accuracy_df = pd.DataFrame(accuracy_results).set_index('Symbol')


        accuracy_df=accuracy_df[(accuracy_df['AVG_Return_No_TP_SL']>0) & (accuracy_df['Accuracy_No_TP_SL']>50)]# & (accuracy_df['Accuracy_No_TP_SL']<accuracy_df['Accuracy_With_TP_SL'])].set_index('Symbol').round(2)
        current_hour = datetime.now(pytz.timezone('Asia/Bangkok'))

        # For Accuracy DataFrame
        accuracy_df['Sell'] = (accuracy_df['Last_Trigger_Date'] + pd.to_timedelta(accuracy_df['Holding_hours'], unit='h')) < current_hour
        # Reorder columns
        desired_columns = [
            'Last_Trigger_Date', 'Holding_hours','Price','TP(%)','SL(%)', 'Sell', 'AVG_Return_No_TP_SL', 
            'Accuracy_No_TP_SL','AVG_Return_With_TP_SL', 
            'Accuracy_With_TP_SL' ,'High_in_x_hours', 'Volume_Increase_Pct', 
            'Num_Signals'
        ]

        accuracy_df = accuracy_df[desired_columns].round(2)
        # Display the tables
        st.dataframe(accuracy_df.sort_values(['Last_Trigger_Date','Accuracy_No_TP_SL'],ascending=False))
        st.subheader(f"Sell Order")
        st.dataframe(accuracy_df[accuracy_df['Sell']==True].sort_values('Last_Trigger_Date',ascending=False))
else:
    st.warning("Strategy results file not found.")

# Section 2: Variance Gamma Price Simulation
st.subheader("Variance Gamma Price Simulation for Middle term trader")

asset = st.text_input("Enter Ticker Symbol (e.g., BTC-USD):", "BTC-USD")
num_scenarios = st.slider("Number of Scenarios:", 100, 5000, 1000)
steps = st.slider("Simulation Days:", 30, 365, 365)

if st.button("Generate Price Simulation"):
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=4)).strftime('%Y-%m-%d')
    price_df = yf.download(asset, start=start_date, end=end_date,progress=False)
    if price_df.empty:
        st.warning(f"No data found for {asset}. Please check if the symbol is valid on Yahoo Finance.")
    else:
        #print(price_df.info())
        price_df.columns = [col[0] for col in price_df.columns]

        #price_df = price_df['Close']
        #print(price_df.head())
        log_return_df = log_return(price_df.dropna())
        #print(log_return_df)
        # Fit Variance Gamma
        log_returns = log_return_df.dropna().values.reshape(-1)

        (c_fit, sigma_fit, theta_fit, nu_fit) = fit_ml(log_returns, maxiter=1000)

        # Simulate Scenarios
        simulated_vg_pct = simulate_vg_scenarios_pct(
            S0=price_df['Close'].iloc[-1],
            c_fit=c_fit,
            sigma_fit=sigma_fit,
            theta_fit=theta_fit,
            nu_fit=nu_fit,
            steps=steps,
            num_scenarios=num_scenarios,
            start_date=end_date
        )

        # Calculate Portfolio Growth

        port_growth = price_df['Close'].iloc[-1] * (1 + simulated_vg_pct).dropna().cumprod()

        # Calculate Percentiles
        cumulative_max = port_growth.cummax()
        cumulative_min = port_growth.cummin()
        monthly_cum_max = cumulative_max.resample('ME').last()
        monthly_cum_min = cumulative_min.resample('ME').last()

        monthly_percentile_extremes = {}
        for month in monthly_cum_max.index:
            median_max = monthly_cum_max.loc[month].median()
            percentile_75_max = monthly_cum_max.loc[month].quantile(0.75)
            median_min = monthly_cum_min.loc[month].median()
            percentile_25_min = monthly_cum_min.loc[month].quantile(0.25)
            monthly_percentile_extremes[month] = {
                "50% Prob. Price Up": median_max,
                "25% Prob. Price Up": percentile_75_max,
                "50% Prob. Price Down": median_min,
                "25% Prob. Price Down": percentile_25_min
            }
        percentile_extremes_df = pd.DataFrame(monthly_percentile_extremes).T


        fig = go.Figure()
        # Get the latest date in the dataset
        latest_date = price_df.index.max()

        # Calculate the date one year ago
        one_year_ago = latest_date - timedelta(days=720)

        # Filter for data within the last year
        price_one_year = price_df[price_df.index >= one_year_ago]
        # Add historical data
        fig.add_trace(go.Scatter(
            x=price_one_year.index, 
            y=price_one_year['Close'], 
            mode='lines', 
            name='Historical Price', 
            line=dict(color='blue')
        ))

        # Add percentile lines
        for col, color, dash in zip(
            ["50% Prob. Price Up", "25% Prob. Price Up", "50% Prob. Price Down", "25% Prob. Price Down"],
            ['green', 'lightgreen', 'red', 'lightcoral'],
            [None, 'dash', None, 'dash']
        ):
            if col in percentile_extremes_df.columns:  # Ensure the column exists
                fig.add_trace(go.Scatter(
                    x=percentile_extremes_df.index, 
                    y=percentile_extremes_df[col], 
                    mode='lines', 
                    name=col, 
                    line=dict(color=color, dash=dash)
                ))

        # Update layout
        fig.update_layout(
            title=f"{asset} Price Simulation with Percentiles",
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Legend",
            template="plotly_white",
            height=600,  # Adjust height for better visualization
            width=1000   # Adjust width for better visualization
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)
