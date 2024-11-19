import streamlit as st
import numpy as np
from scipy import optimize, special
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Variance Gamma functions
def log_return(price_df):
    price_series = price_df['Close']
    log_returns = np.log(price_series / price_series.shift(1)).values.ravel()  # Flatten to 1D
    return pd.DataFrame(log_returns,index=price_df.index)

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

# Streamlit App
st.title("Variance Gamma Price Simulation and Percentiles")
st.markdown("""
    ---
    ### Contact Me
    [Facebook](https://www.facebook.com/profile.php?id=100086804432808).
    """)

# User Inputs
asset = st.text_input("Enter Ticker Symbol (e.g., BTC-USD):", "BTC-USD")
num_scenarios = st.slider("Number of Scenarios:", 100, 5000, 1000)
steps = st.slider("Simulation Days:", 30, 365, 365)
#print("asset",asset)

# Button to Generate Chart
if st.button("Generate Chart"):

    # Fetch Data
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=4)).strftime('%Y-%m-%d')
    price_df = yf.download(asset, start=start_date, end=end_date,progress=False)

    price_df.columns = [col[0] for col in price_df.columns]

    #price_df = price_df['Close']
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
        title="Price Simulation with Percentiles",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_white",
        height=600,  # Adjust height for better visualization
        width=1000   # Adjust width for better visualization
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)