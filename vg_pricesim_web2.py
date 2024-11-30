import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timezone, timedelta
import yfinance as yf
from scipy import optimize
import plotly.graph_objects as go
import pytz
from config import *
from pandas.tseries.offsets import DateOffset

# Directory to save crypto data files
data_dir = "crypto_data"

# Ensure the directory exists
os.makedirs(data_dir, exist_ok=True)

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

    # Determine the data fetch period
    if not data.empty:
        # last_date = data.index.max()

        # # Ensure last_date is timezone-aware (convert to UTC)
        # if last_date.tzinfo is None:
        #     last_date = last_date.tz_localize("UTC")

        # # Check if last_date is more than 1 hour behind current_hour
        # if (current_time - last_date).total_seconds() > 3600:
        #     end_period = '1d'
        # else:
        #     return data  # No update needed
        #No need to check cause data is not complete , need to re update
        end_period = '5d'
    else:
        end_period = '1mo'

    # Fetch updated data
    updated_data = yf.Ticker(ticker).history(period=end_period, interval="1h")
    if updated_data.empty:
        print(f"No new data found for {ticker}.")
        return data

    updated_data = updated_data.rename_axis("timestamp").reset_index()
    updated_data = updated_data[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    updated_data["timestamp"] = pd.to_datetime(updated_data["timestamp"])
    updated_data.set_index("timestamp", inplace=True)

    # Combine old and new data, replace duplicates with new data
    combined_data = pd.concat([data, updated_data]).sort_index()
    combined_data = combined_data[~combined_data.index.duplicated(keep="last")]

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

def find_last_trigger_date_and_price(data, x_days, volume_increase_pct, no_new_order_period=0):
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
    no_new_order_period = int(no_new_order_period)
    if no_new_order_period>0:
    # Suppress triggers within the holding period
        # print(data.head())
        # print(data.info())
        
        trigger_indices = data.index[data['trigger']].to_list()  # Indices of trigger events
        for idx in trigger_indices:
            suppression_end_time = idx + DateOffset(hours=no_new_order_period)

            next_idx_position = data.index.searchsorted(idx) + 1
            if next_idx_position < len(data.index):
                next_idx = data.index[next_idx_position]
            else:
                next_idx = idx 

            # Suppress subsequent triggers within the suppression range
            data.loc[next_idx : suppression_end_time, 'trigger'] = False

    # Find the last trigger date and price
    if data['trigger'].any():
        last_trigger_row = data[data['trigger']].iloc[-1]
        last_trigger_date = last_trigger_row.name
        last_trigger_price = last_trigger_row['Close']
    else:
        last_trigger_date = None
        last_trigger_price = None

    return last_trigger_date, last_trigger_price 



# Streamlit App Layout with Navigation
st.title("Crypto Strategy Performance and Price Simulation")
st.text("Analyze crypto strategies and simulate future price scenarios.")

# Sidebar Navigation
navigation = st.sidebar.radio("Select Page", ["Crypto Alert Signal", "Price Simulation"])

# Current USD/THB Exchange Rate
usd_to_thb_rate = current_USDTHB()

# Navigation: Crypto Alert Signal
if navigation == "Crypto Alert Signal":
    st.subheader("Crypto Strategy Signal Trigger for Short Term Trader")
    param_result_file = "param_result_with_TP_SL.csv"

    if os.path.exists(param_result_file):
        crypto_data = fetch_update_data()
        comparison_df = pd.read_csv(param_result_file, index_col=0)
        if crypto_data is not None:
            accuracy_results = []
            best_df = comparison_df.copy()

            # Iterate through symbols and calculate results
            for symbol, df in crypto_data.items():
                best_accuracy_row = best_df[best_df['Symbol'] == symbol]
                if best_accuracy_row.empty:
                    continue

                # Extract parameters
                best_accuracy_params = {
                    'x_hours': best_accuracy_row['High_in_x_hours'].values[0],
                    'volume_increase_pct': best_accuracy_row['Volume_Increase_Pct'].values[0],
                    'holding_period': best_accuracy_row['Holding_Period'].values[0],
                    'TP(%)': best_accuracy_row['TP(%)'].values[0],
                    'SL(%)': best_accuracy_row['SL(%)'].values[0],
                    'Num_Signals': best_accuracy_row['Num_Signals'].values[0],
                    'Total_Return_No_TP_SL': best_accuracy_row['Total_Return_no_tp_sl'].values[0],
                    'Accuracy_No_TP_SL': best_accuracy_row['Accuracy_no_tp_sl'].values[0],
                    'Total_Return_With_TP_SL': best_accuracy_row['Total_Return_with_tp_sl'].values[0],
                    'Accuracy_With_TP_SL': best_accuracy_row['Accuracy_with_tp_sl'].values[0],
                }

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
                    'Price(THB)': last_trigger_price * usd_to_thb_rate,
                    'Price(USD)': last_trigger_price,
                    'High_in_x_hours': best_accuracy_params['x_hours'],
                    'Volume_Increase_Pct': best_accuracy_params['volume_increase_pct'],
                    'Holding_hours': best_accuracy_params['holding_period'],
                    'TP(%)': best_accuracy_params['TP(%)'],
                    'SL(%)': best_accuracy_params['SL(%)'],
                    'Num_Signals': best_accuracy_params['Num_Signals'],
                    'AVG_Return_No_TP_SL': (
                        best_accuracy_params['Total_Return_No_TP_SL'] /
                        best_accuracy_params['Num_Signals']
                    ) if best_accuracy_params['Num_Signals'] else None,
                    'Accuracy_No_TP_SL': best_accuracy_params['Accuracy_No_TP_SL'],
                    'AVG_Return_With_TP_SL': (
                        best_accuracy_params['Total_Return_With_TP_SL'] /
                        best_accuracy_params['Num_Signals']
                    ) if best_accuracy_params['Num_Signals'] else None,
                    'Accuracy_With_TP_SL': best_accuracy_params['Accuracy_With_TP_SL'],
                })

            # Create DataFrame
            accuracy_df = pd.DataFrame(accuracy_results).set_index('Symbol')

            # Add columns for total portfolio allocation
            total_portfolio_value = st.number_input("Enter Total Portfolio Value (in THB):", value=100000, step=1000)
            accuracy_df['Amount_in_Baht'] = accuracy_df['Price(THB)'] * accuracy_df['Num_Signals']
            accuracy_df['Amount_in_USD'] = accuracy_df['Amount_in_Baht'] / usd_to_thb_rate

            # Display DataFrame
            st.dataframe(accuracy_df.round(2))
    else:
        st.warning("Strategy results file not found.")

# Navigation: Price Simulation
elif navigation == "Price Simulation":
    st.subheader("Variance Gamma Price Simulation for Middle Term Trader")

    asset = st.text_input("Enter Ticker Symbol (e.g., BTC-USD):", "BTC-USD")
    num_scenarios = st.slider("Number of Scenarios:", 100, 5000, 1000)
    steps = st.slider("Simulation Days:", 30, 365, 365)

    if st.button("Generate Price Simulation"):
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=4)).strftime('%Y-%m-%d')
        price_df = yf.download(asset, start=start_date, end=end_date, progress=False)

        if price_df.empty:
            st.warning(f"No data found for {asset}. Please check if the symbol is valid on Yahoo Finance.")
        else:
            log_return_df = log_return(price_df.dropna())
            log_returns = log_return_df.dropna().values.reshape(-1)

            # Fit Variance Gamma
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

            # Plot simulation
            fig = go.Figure()
            for col in simulated_vg_pct.columns:
                fig.add_trace(go.Scatter(x=simulated_vg_pct.index, y=simulated_vg_pct[col], mode='lines'))

            fig.update_layout(
                title=f"{asset} Price Simulation",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white"
            )

            st.plotly_chart(fig)

