from config import *
import pandas as pd
from datetime import datetime,timedelta
import yfinance as yf


# Function to fetch and save data
def fetch_and_save_data(symbols, limit=301):
    """
    Fetch historical data for multiple symbols, save to files, and return as a dictionary of DataFrames.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=limit)

    # Fetch data
    data = yf.download(
        tickers=symbols,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval='1h',
        group_by='ticker',
        auto_adjust=True,
        threads=True
    )

    crypto_data = {}

    # Handle the multi-index columns returned by yfinance
    if isinstance(data.columns, pd.MultiIndex):
        for symbol in symbols:
            if symbol in data.columns.levels[0]:  # Check if symbol has data
                symbol_data = data[symbol].dropna().reset_index()
                symbol_data.columns = ['timestamp'] + list(symbol_data.columns[1:])
                crypto_data[symbol] = symbol_data
                filename = f"{data_dir}/{symbol.replace('-', '_')}.csv"
                symbol_data.to_csv(filename, index=False)
    else:
        # If data does not have MultiIndex, handle it as a single DataFrame
        for symbol in symbols:
            if not data.empty and symbol in data:
                symbol_data = data[[col for col in data.columns if col.startswith(symbol)]].dropna().reset_index()
                symbol_data.columns = ['timestamp'] + list(symbol_data.columns[1:])
                crypto_data[symbol] = symbol_data
                filename = f"{data_dir}/{symbol.replace('-', '_')}.csv"
                symbol_data.to_csv(filename, index=False)
    
    return crypto_data


fetch_and_save_data(crypto_symbols)
