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

# Directory to save crypto data files
data_dir = "crypto_data"

# Ensure the directory exists
os.makedirs(data_dir, exist_ok=True)

# Number of days for filtering last trigger
num_last_trig_day = 3

# List of crypto symbols
crypto_symbols = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'DOGE/USDT',
    'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 'LTC/USDT',
    'SHIB/USDT', 'TRX/USDT', 'AVAX/USDT', 'UNI/USDT', 'LINK/USDT',
    'ATOM/USDT', 'ETC/USDT', 'XLM/USDT', 'BCH/USDT',
    'APT/USDT', 'FIL/USDT', 'ALGO/USDT', 'VET/USDT', 'ICP/USDT',
    'NEAR/USDT', 'QNT/USDT', 'FTM/USDT', 'EOS/USDT', 'SAND/USDT',
    'AAVE/USDT', 'MANA/USDT', 'THETA/USDT', 'XTZ/USDT', 'EGLD/USDT',
    'AXS/USDT', 'CHZ/USDT', 'RPL/USDT', 'GRT/USDT', 'CAKE/USDT',
    'KAVA/USDT', 'ZIL/USDT', 'XEC/USDT', 'BAT/USDT', 'CRV/USDT',
    'DYDX/USDT', 'GALA/USDT', 'STX/USDT', 'IMX/USDT'
]

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
def fetch_and_save_data(symbol, timeframe='1d', limit=100, delay=0.2):
    exchange = ccxt.binance()
    all_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Save to file
    filename = f"{data_dir}/{symbol.replace('/', '_')}.csv"
    df.to_csv(filename, index=False)
    
    # Add delay to respect rate limits
    time.sleep(delay)
    return df

# Check if file is current
def is_file_current(filename):
    if not os.path.exists(filename):
        return False
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    last_date = df['timestamp'].max().date()
    return last_date == datetime.now(timezone.utc).date()

# Load data from files or fetch new data
def load_or_fetch_data(symbols):
    crypto_data = {}
    outdated_symbols = []

    # Check files
    for symbol in symbols:
        filename = f"{data_dir}/{symbol.replace('/', '_')}.csv"
        if is_file_current(filename):
            crypto_data[symbol] = pd.read_csv(filename)  # Load current file
        else:
            outdated_symbols.append(symbol)  # Mark as outdated

    # Show fetch button if there are outdated files
    if outdated_symbols:
        st.warning(f"The following symbols have outdated data: {', '.join(outdated_symbols)}")
        if st.button("Fetch Latest Data"):
            with st.spinner("Fetching data..."):
                progress = st.progress(0)
                for i, symbol in enumerate(outdated_symbols):  # Fetch only outdated symbols
                    crypto_data[symbol] = fetch_and_save_data(symbol)
                    progress.progress((i + 1) / len(outdated_symbols))
                progress.empty()
                st.success("Outdated data fetched successfully!")
            return crypto_data, True
        else:
            st.warning("Please click the button above to fetch the latest data for outdated symbols.")
            return crypto_data, False

    return crypto_data, True

# Function to filter data for the last 7 days
def filter_last_7_days(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df[df['timestamp'] >= (datetime.now(timezone.utc) - pd.Timedelta(days=7))]

# Function to find the last trigger and calculate metrics
def find_last_trigger(data, x_days, volume_increase_pct, holding_period):
    # Calculate X-day high (excluding the current day)
    data['x_day_high'] = data['close'].shift(1).rolling(window=x_days).max()

    # Calculate volume percentage increase
    data['volume_pct_increase'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1) * 100

    # Trigger condition: Price above X-day high and volume increased by X%
    data['trigger'] = (data['close'] > data['x_day_high']) & (data['volume_pct_increase'] > volume_increase_pct)

    # Calculate returns for triggered signals with the holding period
    data['future_close'] = data['close'].shift(-holding_period)  # Close price after the holding period
    data['returns'] = np.where(data['trigger'], (data['future_close'] - data['close']) / data['close'], 0)

    # Metrics
    num_signals = data['trigger'].sum()  # Number of signals
    total_return = data['returns'].sum() * 100  # Total return in percentage
    success_signals = (data['returns'] > 0).sum()  # Number of successful signals
    accuracy = success_signals / num_signals * 100 if num_signals > 0 else 0  # Accuracy percentage
    mean_return = data['returns'][data['returns'] != 0].mean()  # Mean return for signals
    std_dev = data['returns'][data['returns'] != 0].std()  # Standard deviation of returns
    sharpe_ratio = mean_return / std_dev if std_dev != 0 else 0  # Sharpe Ratio

    # Last trigger date
    last_trigger_date = data.loc[data['trigger'], 'timestamp'].max()

    return last_trigger_date, num_signals, total_return, accuracy, sharpe_ratio

# Streamlit App Layout
st.title("Crypto Strategy Performance and Price Simulation")
st.text("Analyze crypto strategies and simulate future price scenarios.")
st.text("Backtest strategy ด้วย ราคาทำ new high ในช่วง X วัน และ Volume ขึ้น Y% ข้อมูลตังแต่ 2023")
st.text("ช่วง Bull run ตอนที่ราคา BTC ATH ,เหรียญอื่นจะขึ้นตาม ช่วงที่หมด Bull run ไม่ควรใช้ strategy นี้ และใช้เพื่อการศึกษาเท่านั้น")
# Section 1: Strategy Analysis (Accuracy and Return Tables)
st.subheader("Crypto Strategy Backtesting")
param_result_file = "param_result.csv"

if os.path.exists(param_result_file):
    crypto_data, data_ready = load_or_fetch_data(crypto_symbols)
    if data_ready:
        accuracy_results = []
        return_results = []
        results_df = pd.read_csv(param_result_file)
        # Process data
        for symbol, df in crypto_data.items():
            # Best accuracy
            best_accuracy_row = results_df[results_df['Symbol'] == symbol].sort_values(by='Accuracy', ascending=False).iloc[0]
            last_trigger_date, num_signals_acc, total_return_acc, accuracy_acc, sharpe_acc = find_last_trigger(
                df.copy(),
                best_accuracy_row['High_in_x_days'],
                best_accuracy_row['Volume_Increase_Pct'],
                best_accuracy_row['Holding_Period']
            )
            accuracy_results.append({
                'Symbol': symbol,
                'Last_Trigger_Date': last_trigger_date,
                'Holding_Days': best_accuracy_row['Holding_Period'],
                'Total_Return': total_return_acc,
                'Accuracy': accuracy_acc,
                'High_in_x_days': best_accuracy_row['High_in_x_days'],
                'Volume_Increase_Pct': best_accuracy_row['Volume_Increase_Pct'],
                'Num_Signals': num_signals_acc,
                'Sharpe_Ratio': sharpe_acc
            })

            # Best return
            best_return_row = results_df[results_df['Symbol'] == symbol].sort_values(by='Total_Return', ascending=False).iloc[0]
            last_trigger_date, num_signals_ret, total_return_ret, accuracy_ret, sharpe_ret = find_last_trigger(
                df.copy(),
                best_return_row['High_in_x_days'],
                best_return_row['Volume_Increase_Pct'],
                best_return_row['Holding_Period']
            )
            return_results.append({
                'Symbol': symbol,
                'Last_Trigger_Date': last_trigger_date,
                'Holding_Days': best_accuracy_row['Holding_Period'],
                'Total_Return': total_return_acc,
                'Accuracy': accuracy_acc,
                'High_in_x_days': best_accuracy_row['High_in_x_days'],
                'Volume_Increase_Pct': best_accuracy_row['Volume_Increase_Pct'],
                'Num_Signals': num_signals_acc,
                'Sharpe_Ratio': sharpe_acc
            })

        # Convert to DataFrames
        accuracy_df = pd.DataFrame(accuracy_results)
        return_df = pd.DataFrame(return_results)
        #print(accuracy_df.head())
        # Ensure 'Last_Trigger_Date' is converted to datetime
        accuracy_df['Last_Trigger_Date'] = pd.to_datetime(accuracy_df['Last_Trigger_Date'], errors='coerce')
        return_df['Last_Trigger_Date'] = pd.to_datetime(return_df['Last_Trigger_Date'], errors='coerce')
        #print(accuracy_df.info())
        # Set the filter date
        
        filter_date = datetime.now(timezone.utc) - pd.Timedelta(days=num_last_trig_day)
        filter_date = filter_date.replace(tzinfo=None)
        #print("filter_date",filter_date)
        # Apply the filter
        accuracy_df = accuracy_df[accuracy_df['Last_Trigger_Date'] >= filter_date]
        return_df = return_df[return_df['Last_Trigger_Date'] >= filter_date]

        accuracy_df=accuracy_df[accuracy_df['Total_Return']>0]
        return_df=return_df[return_df['Total_Return']>0]
        # Display the tables
        st.subheader(f"Return Table (Last {num_last_trig_day} Days)")
        st.dataframe(return_df.sort_values('Last_Trigger_Date',ascending=False))

        st.subheader(f"Accuracy Table (Last {num_last_trig_day} Days)")
        st.dataframe(accuracy_df.sort_values('Last_Trigger_Date',ascending=False))
else:
    st.warning("Strategy results file not found.")

# Section 2: Variance Gamma Price Simulation
st.subheader("Variance Gamma Price Simulation")

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
