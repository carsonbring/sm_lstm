import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from data import loader


def read_data(stock_ticker, db_path='stocks.db'):
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve data from the database
    cursor.execute(f"""
        SELECT 
            Date, Adj_Close, Volume, EPS_TTM, PE_Ratio, BB_Middle, BB_Upper, BB_Lower, 
            MACD, MACD_Signal, MACD_Diff, MFI, Stoch_Osc, Stoch_Osc_Signal
        FROM {stock_ticker} ORDER BY Date ASC
    """)  
    data = cursor.fetchall()
    if not data:
        print(f"No data found in the database. Please create the table for {stock_ticker} first.")
    # Close the connection
    conn.close()
    return data


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
            new_df = loader.obtain_stock_data(stock_ticker, start_date, end_date)

            # Add new records to the table and remove the first row for each new record added
            for index, row in new_df.iterrows():
                cursor.execute(f"""
                                SELECT * FROM {stock_ticker} WHERE Date = ?
                            """, (row['Date'].strftime('%Y-%m-%d'),))

                existing_data = cursor.fetchone()
                if not existing_data:
                    new_records_added = True
                    cursor.execute(f"""
                                    INSERT INTO {stock_ticker} (Date, Open, High, Low, Close, Adj_Close, Volume, EPS_TTM, PE_Ratio, BB_Middle, BB_Upper, BB_Lower, MACD, MACD_Signal, MACD_Diff, MFI, Stoch_Osc, Stoch_Osc_Signal)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (
                        row['Date'].strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'],
                        row['Volume'], row['EPS_TTM'], row['PE_Ratio'], row['BB_Middle'], row['BB_Upper'], row['BB_Lower'],
                        row['MACD'], row['MACD_Signal'], row['MACD_Diff'], row['MFI'], row['Stoch_Osc'], row['Stoch_Osc_Signal']))
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
                EPS_TTM REAL,
                PE_Ratio REAL,
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
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
        df = loader.obtain_stock_data(stock_ticker, start_date, end_date)

        # Insert data into the table
        for index, row in df.iterrows():
            cursor.execute(f"""
                INSERT INTO {stock_ticker} (Date, Open, High, Low, Close, Adj_Close, Volume, EPS_TTM, PE_Ratio, BB_Middle, BB_Upper, BB_Lower, MACD, MACD_Signal, MACD_Diff, MFI, Stoch_Osc, Stoch_Osc_Signal) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (row['Date'].strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'],
                  row['Volume'], row['EPS_TTM'], row['PE_Ratio'], row['BB_Middle'], row['BB_Upper'], row['BB_Lower'],
                    row['MACD'], row['MACD_Signal'], row['MACD_Diff'], row['MFI'], row['Stoch_Osc'], row['Stoch_Osc_Signal']))

        # Commit changes to the database
        conn.commit()
    # Close the connection
    
    conn.close()

