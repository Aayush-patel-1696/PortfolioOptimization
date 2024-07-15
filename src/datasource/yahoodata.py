import warnings

import pandas as pd
import yfinance as yf



class YahooDataSource:

    def __init__(self,start_date,end_date,tickers,columns,interval):
        self.tickers = tickers
        self.columns = columns
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data  = self.get_yahoo_data()

    def get_yahoo_data(self):

        data = {}
        for symbol in self.tickers:
            
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=self.start_date, end=self.end_date,interval=self.interval)
                hist.reset_index(inplace=True)
                if not hist.empty:
                    for col in self.columns:
                        data[symbol + "_" + col] = hist[col].to_numpy()
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

        return data
    
    def get_data_by_column_tickers(self,columns=-1,tickers=-1):
        
        all_tickers = self.tickers
        all_columns = self.columns

        if columns ==-1:
            columns = all_columns
        
        if tickers == -1:
            tickers = all_tickers
        
        validated_tickers = set(tickers).intersection(all_tickers)
        validated_columns = set(columns).intersection(all_columns)

        if len(set(tickers)) != len(set(validated_tickers)):
            warnings.warn(f"Following Tickers are not Found {set(tickers)-set(validated_tickers)}")
        
        if len(set(columns)) != len(set(validated_columns)):
            warnings.warn(f"Following Columns are not Found {set(columns)-set(validated_columns)}")
        
        ticker_columns = self.create_ticker_columns(validated_columns,validated_tickers)

        return pd.DataFrame({key: self.data[key] for key in ticker_columns})
    
    def create_ticker_columns(self,columns,tickers):

        ticker_columns = []
        for tick in tickers:
            for col in columns:
                name = tick+"_"+col
                ticker_columns.append(name)

        return ticker_columns
    
    def get_tickers(self,ticker_columns):

        return [i.split("_")[0] for i in ticker_columns]

    
