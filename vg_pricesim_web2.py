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
    
    crypto_data = {}
    file_list = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    total_files = len(file_list)

    # Add a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file_name in enumerate(file_list):
        ticker = file_name.replace("_", "-").replace(".csv", "")  # Convert file name to ticker
        file_path = os.path.join(data_dir, file_name)
        status_text.write(f"Processing {file_name} ({i + 1}/{total_files})...")

        # Update progress bar
        progress_bar.progress((i + 1) / total_files)

        # Update crypto data
        updated_df = update_crypto_data(file_path, ticker)
        crypto_data[ticker] = updated_df  # Add updated DataFrame to the dictionary

    # Clear the progress and status
    progress_bar.empty()
    
    st.session_state.updated_data=True  # Set a default value
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
st.text("Analyze crypto strategies")

# Sidebar Navigation
navigation = st.sidebar.radio("Select Page", ["Crypto Alert Signal", "Price Simulation"])

# Current USD/THB Exchange Rate
usd_to_thb_rate = current_USDTHB()

# Navigation: Crypto Alert Signal
if navigation == "Crypto Alert Signal":
    st.subheader("Crypto Strategy Signal Trigger for Short Term Trader")
    param_result_file = "param_result_with_TP_SL.csv"
    #accuracy_df.to_csv('accuracy_df.csv')
    if "total_portfolio" not in st.session_state:
        st.session_state.total_portfolio = 100000  # Set a default value

    st.session_state.total_portfolio = st.number_input(
        "Enter Total Portfolio Value (in THB):", 
        value=st.session_state.total_portfolio,  # Initialize with session state value
        step=1000, 
        key="total_portfolio_input"
    )
    if "updated_data" not in st.session_state: 
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

                    # Extract parameters safely
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
                        'Price(USD)': last_trigger_price,
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
                        'Weight':best_accuracy_row['Weight'].values[0]
                    })

                # Create DataFrame
                accuracy_df = pd.DataFrame(accuracy_results).set_index('Symbol')

                current_hour = datetime.now(pytz.timezone('Asia/Bangkok'))

                # For Accuracy DataFrame
                accuracy_df['Sell'] = (accuracy_df['Last_Trigger_Date'] + pd.to_timedelta(accuracy_df['Holding_hours'], unit='h')) < current_hour
                # Reorder columns

            st.session_state.accuracy_df=accuracy_df
        else:
            st.warning("Strategy results file not found.")
    
    desired_columns = [
    'Last_Trigger_Date', 'Holding_hours','Price(THB)','Price(USD)','Weight','Amount_in_Baht','Amount_in_USD','TP(%)','SL(%)', 'Sell', 'AVG_Return_No_TP_SL', 
    'Accuracy_No_TP_SL','AVG_Return_With_TP_SL', 
    'Accuracy_With_TP_SL' ,'High_in_x_hours', 'Volume_Increase_Pct', 
    'Num_Signals'
    ]
    st.session_state.accuracy_df['Amount_in_Baht'] = st.session_state.accuracy_df['Weight'] * st.session_state.total_portfolio 
    st.session_state.accuracy_df['Amount_in_USD'] = st.session_state.accuracy_df['Amount_in_Baht'] / usd_to_thb_rate
    st.session_state.accuracy_df = st.session_state.accuracy_df[desired_columns].round(4)

    #st.text("Start Port Value:",st.session_state.total_portfolio,"Baht")
    st.write(f"Start Port Value: {st.session_state.total_portfolio} baht")
    # Display the tables
    st.subheader(f"Buy Order")
    st.dataframe(st.session_state.accuracy_df[st.session_state.accuracy_df['Sell']==False].sort_values(['Last_Trigger_Date','Accuracy_No_TP_SL'],ascending=False))
    st.subheader(f"Sell Order")
    st.dataframe(st.session_state.accuracy_df[st.session_state.accuracy_df['Sell']==True].sort_values('Last_Trigger_Date',ascending=False))

# Navigation: Price Simulation
elif navigation == "Price Simulation":
    st.subheader("Variance Gamma Price Simulation for Middle Term Trader")

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

            # Get the last row from simulated_vg_pct
            last_row = simulated_vg_pct.iloc[-1]

            # Find the median value
            last_value_50_index = last_row.median()

            # Find the index of the value closest to the median
            that_idx = (last_row - last_value_50_index).abs().idxmin()

            # Calculate the 50% price percentile using the index
            price_50 = price_df['Close'].iloc[-1] * (1 + simulated_vg_pct[that_idx])

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

            fig.add_trace(go.Scatter(
                x=percentile_extremes_df.index, 
                y=price_50, 
                mode='lines', 
                name='Sample Price', 
                line=dict(color='orange',dash='dash')
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

            # fig.add_annotation(
            #     x=simulated_vg_pct.index[-1],  # Position at the end of the x-axis
            #     y=price_50,  # Position at the 50% price percentile value
            #     text=f"50% Price Percentile: {price_50:.2f}",
            #     showarrow=True,
            #     arrowhead=2,
            #     ax=50,  # Horizontal offset for arrow
            #     ay=0,   # Vertical offset for arrow
            #     font=dict(color="purple")
            # )

            # Update layout
            fig.update_layout(
                title=f"{asset} Price Simulation with Percentiles",
                xaxis_title="Date",
                yaxis_title="Price",
                legend_title="Legend",
                template="plotly_white",
                height=600,
                width=1000
            )
            # Display the chart in Streamlit
            st.plotly_chart(fig)

