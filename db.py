import sqlite3
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import ta
import requests
from typing import Union


def get_earnings_histo(symbol: str) -> Union[None, pd.DataFrame]:
    symbol = symbol.upper()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    web_request = requests.get(url=f'https://finance.yahoo.com/calendar/earnings?symbol={symbol}', headers=headers)

    try:
        web_tables = pd.read_html(web_request.text)
        results_df = web_tables[0]
    except Exception as ex:
        print(f"No earnings histo found for ticker: {symbol}")
        print(f"Error is: {ex}")
        return None
    if len(results_df) == 0:
        print(f"No earnings histo found for ticker: {symbol}")
        return None
    dates_df = results_df['Earnings Date'].str.split(',', expand=True, n=2)[[0, 1]]
    results_df['earnings_date'] = dates_df[0].astype(str) + ' ' + dates_df[1].astype(str)
    results_df['earnings_date'] = pd.to_datetime(results_df['earnings_date'])

    return results_df


def retrieve_data(stock_ticker, db_path='stocks.db'):
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve data from the database
    cursor.execute(f"SELECT Date, Adj_Close FROM {stock_ticker} ORDER BY Date ASC")
    data = cursor.fetchall()
    if not data:
        print(f"No data found in the database. Please create the table for {stock_ticker} first.")
    # Close the connection
    conn.close()
    return data


def prepare_data(data, seq_length):
    # Convert data to numpy array
    data_array = np.array(data)
    print(data_array)
    print("-------------------")
    # Extract features and target variable
    # I need to retrieve some more useful features before I get more into the LSTM development
    features = data_array[:, 1].astype(np.float32)

    print(features)
    # Normalize features
    min_val = np.min(features, axis=0)
    max_val = np.max(features, axis=0)
    features = (features - min_val) / (max_val - min_val)

    # Convert data into sequences
    sequences = []
    targets = []
    for i in range(len(features) - seq_length):
        sequences.append(features[i:i + seq_length])
        targets.append(features[i + seq_length])

    sequences = np.array(sequences)
    targets = np.array(targets)
    # Convert sequences and targets to PyTorch tensors
    sequences_tensor = torch.tensor(sequences)
    targets_tensor = torch.tensor(targets)

    return sequences_tensor, targets_tensor


def create_data_loaders(sequences, targets, batch_size):
    # Create TensorDataset
    dataset = TensorDataset(sequences, targets)
    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def update_stock_table(stock_ticker, db_path='stocks.db'):
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    new_records_added = False
    # Get the last row from the table
    cursor.execute(f"SELECT * FROM {stock_ticker} ORDER BY Date DESC LIMIT 1")
    last_row = cursor.fetchone()

    if last_row:
        last_date = datetime.strptime(last_row[0], '%Y-%m-%d')
        today = datetime.now()

        # Determine the date threshold based on the current day of the week
        if today.weekday() == 6:  # Sunday
            date_threshold = today - timedelta(days=2)
        elif today.weekday() == 5:  # Saturday
            date_threshold = today - timedelta(days=1)
        else:
            date_threshold = today - timedelta(days=1)

        # Fetch new data if the last recorded date is before the threshold date
        if last_date < date_threshold:
            # Fetch new data from Yahoo Finance
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            new_df = yf.download(stock_ticker, start=start_date, end=end_date)

            # Add new records to the table and remove the first row for each new record added
            for index, row in new_df.iterrows():
                cursor.execute(f"""
                                SELECT * FROM {stock_ticker} WHERE Date = ?
                            """, (index.strftime('%Y-%m-%d'),))
                existing_data = cursor.fetchone()
                if not existing_data:
                    new_records_added = True
                    cursor.execute(f"""
                                    INSERT INTO {stock_ticker} (Date, Open, High, Low, Close, Adj_Close, Volume)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (
                        index.strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'],
                        row['Volume']))
                    # Remove the first row for each new record added
                    cursor.execute(f"""
                                DELETE FROM {stock_ticker} 
                                WHERE Date IN (
                                    SELECT Date FROM {stock_ticker} ORDER BY Date ASC LIMIT 1
                                )
                                """)

            # Commit changes to the database
            conn.commit()
            if new_records_added:
                print("Database updated successfully.")
            else:
                print("No new data found to update the database.")
        else:
            print("No update needed. Database is up to date.")
    else:
        print("No existing data found in the database.")

    # Close the connection
    conn.close()

def obtain_stock_data(stock_ticker):

    # Fetch the last 30 days of prices using Yahoo Finance
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    df = yf.download(stock_ticker, start=start_date, end=end_date)
    eps_df = get_earnings_histo(stock_ticker)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    if eps_df is not None:
        eps_df['Date'] = pd.to_datetime(eps_df['earnings_date'])
        eps_df = eps_df.sort_values('Date')
        df['EPS'] = pd.merge_asof(df.sort_values('Date'), eps_df, on='Date', direction='backward')['Reported EPS']
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    indicator_bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Middle'] = indicator_bb.bollinger_mavg()
    df['BB_Upper'] = indicator_bb.bollinger_hband()
    df['BB_Lower'] = indicator_bb.bollinger_lband()

    indicator_macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = indicator_macd.macd()
    df['MACD_Signal'] = indicator_macd.macd_signal()
    df['MACD_Diff'] = indicator_macd.macd_diff()

    df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()

    indicator_so = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stock_Osc'] = indicator_so.stoch()
    df['Stock_Osc_Signal'] = indicator_so.stoch_signal()
    return df


def create_stock_table_if_not_exists(stock_ticker, db_path='stocks.db'):
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the table exists
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{stock_ticker}'")
    table_exists = cursor.fetchone()

    if not table_exists:
        # Create table if it doesn't exist
        cursor.execute(f"""
            CREATE TABLE {stock_ticker} (
                Date TEXT,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Adj_Close REAL,
                Volume INTEGER,
                RSI REAL,
                BB_Middle REAL,
                BB_Upper REAL,
                BB_Lower REAL,
                MACD REAL,
                MACD_Signal REAL,
                MACD_Diff REAL,
                MFI REAL,
                Stoch_Osc REAL,
                Stoch_Osc_Signal REAL
                
            )
        """)
        df = obtain_stock_data(stock_ticker)

        # Insert data into the table
        for index, row in df.iterrows():
            cursor.execute(f"""
                INSERT INTO {stock_ticker} (Date, Open, High, Low, Close, Adj_Close, Volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (index.strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'],
                  row['Volume']))

        # Commit changes to the database
        conn.commit()

    # Close the connection
    conn.close()

# prepare_data(retrieve_data('AAPL'), 10)
# Set option to display all columns
pd.set_option('display.max_columns', None)

# Set option to display all rows
pd.set_option('display.max_rows', None)

# Increase the maximum width of each column
pd.set_option('display.max_colwidth', None)

# Optionally, set a large width for the terminal display to avoid wrapping
pd.set_option('display.width', 1000)

#print(obtain_stock_data_yfinance('AAPL'))

print(obtain_stock_data('AAPL'))