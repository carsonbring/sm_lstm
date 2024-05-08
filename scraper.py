import requests
from bs4 import BeautifulSoup

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


print(get_redirect_url("AAPL"))

print(get_PE_ratio("AAPL"))