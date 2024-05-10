import requests
from bs4 import BeautifulSoup
import pandas as pd
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


def get_redirect_url(stock_ticker):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        url = f"https://www.macrotrends.net/stocks/charts/{stock_ticker}"
        response = requests.get(url, headers=headers,  allow_redirects=True)
        redirect = response.url
        return redirect
    except requests.exceptions.RequestException as e:
        print(f"Getting redirect URL failed: {e}")
        return None

def get_PE_ratio(stock_ticker):
    try:
        redirect_url = f"{get_redirect_url(stock_ticker)}pe-ratio"
        response = requests.get(redirect_url).text
        return response
        soup = BeautifulSoup(response, 'html.parser')
        return soup
        pe_ratio = soup.find('table', class_='table')
        return pe_ratio
    except requests.exceptions.RequestException as e:
        print(f"Getting PE ratio failed: {e}")
        return None
