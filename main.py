import pandas as pd 
from data import create_stock_table_if_not_exists, read_data
from data.loader import prepare_data, create_data_loaders
from model.train import train, evaluate


# Set option to display all columns
pd.set_option('display.max_columns', None)

# Set option to display all rows
pd.set_option('display.max_rows', None)

# Increase the maximum width of each column
pd.set_option('display.max_colwidth', None)

# Optionally, set a large width for the terminal display to avoid wrapping
pd.set_option('display.width', 1000)

# update_stock_table('AAPL')

#print(obtain_stock_data_yfinance('AAPL'))
# print(scraper.get_earnings_histo('AAPL'))
# print('----------------------------------------')
# print(utils.obtain_stock_data('AAPL'))
# create_stock_table_if_not_exists('AMD')
#
create_stock_table_if_not_exists('AAPL')
data = read_data('AAPL')
print(data)
train(data)

evaluate(data)
