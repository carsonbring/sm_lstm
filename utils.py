import pandas as pd

def calculate_ttm_eps(current_date, eps_data):
    # Filter reports that occurred before the current date
    past_eps = eps_data[eps_data['Date'] <= current_date].copy()
    past_eps.loc[:, 'Reported EPS'] = pd.to_numeric(past_eps['Reported EPS'], errors='coerce')
    # Get the last four entries or less if not available
    last_four_eps = past_eps.tail(4)

    # Sum their EPS values to get the TTM EPS
    return last_four_eps['Reported EPS'].sum()
