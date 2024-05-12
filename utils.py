import pandas as pd
import scraper
import yfinance as yf
import ta
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np


def calculate_ttm_eps(current_date, eps_data):
    # Filter reports that occurred before the current date
    past_eps = eps_data[eps_data['Date'] <= current_date].copy()
    past_eps.loc[:, 'Reported EPS'] = pd.to_numeric(past_eps['Reported EPS'], errors='coerce')
    # Get the last four entries or fewer if not available
    last_four_eps = past_eps.tail(4)

    # Sum their EPS values to get the TTM EPS
    return last_four_eps['Reported EPS'].sum()


def obtain_stock_data(stock_ticker, start_date, end_date):
    # Fetch the last 30 days of prices using Yahoo Finance
    df = yf.download(stock_ticker, start=start_date, end=end_date)
    eps_df = scraper.get_earnings_histo(stock_ticker)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    if eps_df is not None:

        eps_df['Date'] = pd.to_datetime(eps_df['earnings_date'])
        eps_df = eps_df.sort_values('Date')

        eps_df = eps_df[eps_df['Reported EPS'] != '-']

        df['EPS_TTM'] = df['EPS_TTM'] = df['Date'].apply(lambda x: calculate_ttm_eps(x, eps_df))

        df['PE_Ratio'] = df['Close'] / df['EPS_TTM']

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
    df['Stoch_Osc'] = indicator_so.stoch()
    df['Stoch_Osc_Signal'] = indicator_so.stoch_signal()
    df = df.dropna()
    return df


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

