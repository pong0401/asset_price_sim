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
import math

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


def fetch_update_data(comparison_df):
    
    crypto_data = {}
    file_list = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    total_files = len(file_list)

    # Add a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, row in comparison_df.iterrows():
        filename = row['Symbol'].replace("-", "_")+'.csv'
        file_path = os.path.join(data_dir, filename)
        status_text.write(f"Processing {row['Symbol']} ({i + 1}/{len(comparison_df)})...")

        # Update progress bar
        progress_bar.progress((i + 1) / len(comparison_df))

        # Update crypto data
        updated_df = update_crypto_data(file_path, row['Symbol'])
        crypto_data[row['Symbol']] = updated_df  # Add updated DataFrame to the dictionary

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
navigation = st.sidebar.radio("Select Page", ["Crypto Alert Signal", "Price Simulation","Reverse DCA"])

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
    start_update=st.button("Start Data Update")
    if start_update:
        if "updated_data" not in st.session_state: 
            if os.path.exists(param_result_file):
                comparison_df = pd.read_csv(param_result_file, index_col=0)
                crypto_data = fetch_update_data(comparison_df)
                
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
    st.session_state.steps=365
    asset = st.text_input("Enter Ticker Symbol (e.g., BTC-USD):", "BTC-USD")
    asset_data_duration=st.number_input("Use data period to calcalte Varaince Gamma :",min_value=1,max_value=4,value=4 )
    num_scenarios = st.slider("Number of Scenarios:", 100, 10000, 1000)
    steps = st.slider("Simulation Days:", 90, 1460, st.session_state.steps)
    apply_4_year_cycle = st.checkbox("Apply 4-Year Cycle Filter", value=False)
    st.subheader("Set Filter for Asset Price Ranges")
    price_min_filter = st.number_input("1 year Minimum Price :", value=0.0)
    price_max_filter = st.number_input("1 year Maximum Price :", value=1000000.0)

    if st.button("Generate Price Simulation"):
        if apply_4_year_cycle:
            steps=1460
            st.session_state.steps=steps

        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=asset_data_duration)).strftime('%Y-%m-%d')
        price_df = yf.download(asset, start=start_date, end=end_date, progress=False)
        if price_df.empty:
            st.warning(f"No data found for {asset}. Please check if the symbol is valid on Yahoo Finance.")
        else:
            log_return_df = log_return(price_df.dropna())
            log_returns = log_return_df.dropna().values.reshape(-1)

            (c_fit, sigma_fit, theta_fit, nu_fit) = fit_ml(log_returns, maxiter=1000)

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

            # Calculate cumulative product for each scenario
            last_asset_price=price_df['Close'].iloc[-1].values[0]
            simulated_prices = (1 + simulated_vg_pct).cumprod() * last_asset_price

            # Calculate min and max prices for the first year
            first_year_prices = simulated_prices.iloc[:365]  # Approx. 1 year (365 trading days)
            yearly_min_prices = first_year_prices.min()
            yearly_max_prices = first_year_prices.max()
            if price_max_filter>yearly_max_prices.max():
                price_max_filter=yearly_max_prices.quantile(0.77)
            #print(yearly_min_prices)
            # Filter scenarios based on user input for price range
            valid_scenarios = (yearly_min_prices >= price_min_filter) & (yearly_max_prices <= price_max_filter)
            simulated_prices = simulated_prices.loc[:, valid_scenarios]
            
            

            if apply_4_year_cycle:

                # Calculate yearly returns
                yearly_returns = simulated_prices.resample('YE').last().pct_change().dropna()
                #print(yearly_returns)
                yearly_returns.index = yearly_returns.index.year  # Use years as index

                # Determine mod-4 values for years
                mod_4_years = yearly_returns.index % 4

                # Filter for years mod 4 == 0, 1, 3 (positive returns expected)
                positive_years = yearly_returns.loc[mod_4_years.isin([0, 1, 3])]
                #print(positive_years)
                # Find scenarios where all relevant years have positive returns
                valid_scenarios_positive = positive_years.columns[(positive_years > 0.2).all(axis=0)]
                #print(valid_scenarios_positive)
                # Filter for years mod 4 == 2 (negative returns expected)
                negative_years = yearly_returns.loc[mod_4_years == 2]

                # Find scenarios where all relevant years have negative returns
                valid_scenarios_negative = negative_years.columns[(negative_years < -0.3).all(axis=0)]
                #print(valid_scenarios_negative)
                # Combine valid scenarios from positive and negative filters
                valid_scenarios = valid_scenarios_positive.intersection(valid_scenarios_negative)
                #print(valid_scenarios)
                # Filter the simulated_prices DataFrame
                simulated_prices = simulated_prices[valid_scenarios]
            #print(simulated_prices)
            last_prices = simulated_prices.iloc[-1]

            # Define 10th percentiles
            percentiles = [0.1 * i for i in range(1, 10)]

            # Find scenarios corresponding to these percentiles
            selected_scenarios = {}
            for p in percentiles:
                percentile_value = last_prices.quantile(p)  # Get the value at the p-th percentile
                closest_idx = (last_prices - percentile_value).abs().idxmin()  # Find the closest scenario
                selected_scenarios[f"{int(p * 100)}th Percentile"] = simulated_prices[closest_idx]

            # Define time horizons based on the simulation steps
            max_horizon = steps  # Maximum prediction horizon based on the slider
            horizon_step = 90    # Define step interval for horizons (e.g., 90 days)
            time_horizons = list(range(horizon_step, max_horizon + 1, horizon_step))

            # Define price step size dynamically based on the price magnitude

            def get_step_size(price):
                if price <= 0:
                    return 0.01  # Handle edge case where price is 0 or negative
                magnitude = math.floor(math.log10(price))
                if magnitude < 0:
                    return 10 ** magnitude  # For fractional prices, step size is 10^magnitude
                return 10 ** (magnitude-1)  # For whole prices, step size is 10^(magnitude-1)

            def format_for_streamlit(df, price_tiers):
                # Sort the DataFrame by numeric index first
                df = df.sort_index(ascending=False)

                # Format the price tiers (index) for display
                formatted_index = df.index.map(lambda x: f"{x:,.4f}" if x < 1 else f"{int(x):,}")
                df.index = formatted_index  # Replace the index with formatted strings

                # Format the percent values (table values) with 2 decimal places
                df = df.map(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
                return df

            # Calculate step size
            step_size = get_step_size(last_asset_price)

            # Create price tiers
            price_tiers = np.arange(
                np.floor(price_min_filter / step_size) * step_size,
                np.ceil(price_max_filter / step_size) * step_size + step_size,
                step_size
            )

            # Initialize the probability table
            probability_table = pd.DataFrame(index=price_tiers)

            # Get the last date in the dataset
            last_date = price_df.index[-1]

            # Calculate probabilities for each time horizon
            for horizon in time_horizons:
                if horizon <= simulated_prices.shape[0]:  # Ensure the horizon is within simulation steps
                    horizon_prices = simulated_prices.iloc[horizon - 1]  # Get prices for the specific horizon
                    horizon_date = (last_date + pd.Timedelta(days=horizon)).strftime('%Y-%m-%d')  # Calculate horizon date
                    column_name = f"{horizon_date}"
                    
                    # Vectorized probability calculations for price tiers
                    probs_above = (horizon_prices.values[:, None] > price_tiers).mean(axis=0)  # For tiers above last price
                    probs_below = (horizon_prices.values[:, None] < price_tiers).mean(axis=0)  # For tiers below last price
                    
                    probabilities = np.where(price_tiers > last_asset_price, probs_above, probs_below)
                    probability_table[column_name] = probabilities

            # Format the table based on price tier scale
            formatted_table = format_for_streamlit(probability_table * 100, price_tiers)

            st.write(f"Probability (%) Table for {asset} (Filtered by Price Range for Multiple Horizons):")
            st.table(formatted_table)

            # Plot the selected scenarios
            fig = go.Figure()
            st.session_state.selected_scenarios=selected_scenarios
            # Add selected scenarios to the chart
            for label, scenario in selected_scenarios.items():
                fig.add_trace(go.Scatter(
                    x=scenario.index,
                    y=scenario,
                    mode='lines',
                    name=label
                ))

            # Update layout
            fig.update_layout(
                title=f"{asset} Price Simulation with Selected Scenarios",
                xaxis_title="Date",
                yaxis_title="Price",
                legend_title="Legend",
                template="plotly_white",
                height=600,
                width=1000
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig)

elif navigation == "Reverse DCA":
    st.subheader("Reverse DCA Results")
    if "selected_scenarios" not in st.session_state:
        st.warning("Please generate Price Simulation first!")
    else:
        def reverse_dca(selected_scenarios):
            reverse_dca_results = {}
            num_days = len(next(iter(selected_scenarios.values())))  # All scenarios have the same duration
            daily_btc_sale = 1 / num_days  # Amount of BTC sold per day

            for label, prices in selected_scenarios.items():
                usd_balance = 0
                btc_holdings = 1
                usd_balances = []

                # Simulate daily sales
                for price in prices:
                    usd_balance += daily_btc_sale * price
                    btc_holdings -= daily_btc_sale
                    usd_balances.append(usd_balance)

                # Store results for the scenario
                reverse_dca_results[label] = pd.Series(usd_balances, index=prices.index)

            return reverse_dca_results, num_days, daily_btc_sale

        # Perform Reverse DCA
        reverse_dca_results, num_days, daily_btc_sale = reverse_dca(st.session_state.selected_scenarios)

        # Display Sale Details
        st.write("### Reverse DCA Sale Details")
        st.write(
            f"Selling **{daily_btc_sale:.6f} BTC per day** over **{num_days} days** for all scenarios."
        )

        # Plot Reverse DCA Results
        reverse_dca_fig = go.Figure()
        for label, usd_balances in reverse_dca_results.items():
            reverse_dca_fig.add_trace(go.Scatter(x=usd_balances.index, y=usd_balances.values, mode='lines', name=f"Reverse DCA {label}"))

        reverse_dca_fig.update_layout(
            title="Reverse DCA Results for Selected Scenarios",
            xaxis_title="Date",
            yaxis_title="USD Balance",
            legend_title="Legend",
            template="plotly_white",
            height=600,
            width=1000
        )
        st.plotly_chart(reverse_dca_fig)







