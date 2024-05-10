import pandas as pd
import scraper
import yfinance as yf
import ta
from datetime import datetime, timedelta


def calculate_ttm_eps(current_date, eps_data):
    # Filter reports that occurred before the current date
    past_eps = eps_data[eps_data['Date'] <= current_date].copy()
    past_eps.loc[:, 'Reported EPS'] = pd.to_numeric(past_eps['Reported EPS'], errors='coerce')
    # Get the last four entries or less if not available
    last_four_eps = past_eps.tail(4)

    # Sum their EPS values to get the TTM EPS
    return last_four_eps['Reported EPS'].sum()


def obtain_stock_data(stock_ticker):
    # Fetch the last 30 days of prices using Yahoo Finance
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
    df = yf.download(stock_ticker, start=start_date, end=end_date)
    eps_df = scraper.get_earnings_histo(stock_ticker)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    if eps_df is not None:

        eps_df['Date'] = pd.to_datetime(eps_df['earnings_date'])
        eps_df = eps_df.sort_values('Date')

        eps_df = eps_df[eps_df['Reported EPS'] != '-']

        df['EPS_TTM'] = df['EPS_TTM'] = df['Date'].apply(lambda x: utils.calculate_ttm_eps(x, eps_df))

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
    df['Stock_Osc'] = indicator_so.stoch()
    df['Stock_Osc_Signal'] = indicator_so.stoch_signal()
    return df

