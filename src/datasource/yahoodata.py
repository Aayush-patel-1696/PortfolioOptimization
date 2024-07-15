import numpy as np
import pandas as pd
import yfinance as yf
import random
import warnings
from concurrent.futures import ThreadPoolExecutor


class YahooDataSource:

    def __init__(self, tickers, start_date, end_date, columns, interval="1d"):
        self.columns = columns
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data = {}
        self.tickers = []
        self.add_tickers(tickers)

    def get_yahoo_data(self, ticker):
        try:
            # hist = yf.download(ticker, start=self.start_date, end=self.end_date, interval=self.interval)
            # hist["Return"] = hist[self.columns].pct_change()
            data = yf.Ticker(ticker)
            hist = data.history(
                start=self.start_date, end=self.end_date, interval=self.interval
            )
            hist["Return"] = hist["Close"].pct_change()
            hist = hist.dropna()
            # hist.reset_index(inplace=True)
            # if not hist.empty:
            #     for col in self.columns:
            #         data[symbol + "_" + col] = hist[col].to_numpy()
            return hist
        except Exception as e:
            raise e

    def add_ticker(self, ticker):
        print(f"Adding {ticker} to the data source")
        if ticker in self.tickers:
            warnings.warn(f"{ticker} already exists in the data source")
            return

        try:
            print(f"Getting data for {ticker}")
            self.tickers.append(ticker)
            data = self.get_yahoo_data(ticker)
            self.data[ticker] = data
        except Exception as e:
            warnings.warn(f"Failed to get data for {ticker}")
            self.tickers.remove(ticker)

    def add_tickers(self, tickers):
        with ThreadPoolExecutor() as executor:
            executor.map(self.add_ticker, tickers)

        # for ticker in tickers:
        #     self.add_ticker(ticker)

    def get_rand_returns(self, n=30):
        if n < 1:
            warnings.warn("Number of days should be greater than 0")
            return

        if n >= self.data[self.tickers[0]].shape[0]:
            warnings.warn("Number of days is greater than the available data")
            return

        if len(self.tickers) == 0:
            warnings.warn("No tickers available")
            return

        data = {}
        random_indexes = np.random.choice(
            self.data[self.tickers[0]].index, n, replace=False
        )
        for ticker in self.tickers:
            data[ticker] = self.data[ticker].loc[random_indexes, self.columns]

        # convert to dataframe
        data = pd.DataFrame(
            {key: data[key].values.flatten() for key in data}, index=random_indexes
        )

        return data

    def get_data_by_column_tickers(self, columns=-1, tickers=-1):

        all_tickers = self.tickers
        all_columns = self.columns

        if columns == -1:
            columns = all_columns

        if tickers == -1:
            tickers = all_tickers

        validated_tickers = set(tickers).intersection(all_tickers)
        validated_columns = set(columns).intersection(all_columns)

        if len(set(tickers)) != len(set(validated_tickers)):
            warnings.warn(
                f"Following Tickers are not Found {set(tickers)-set(validated_tickers)}"
            )

        if len(set(columns)) != len(set(validated_columns)):
            warnings.warn(
                f"Following Columns are not Found {set(columns)-set(validated_columns)}"
            )

        ticker_columns = self.create_ticker_columns(
            validated_columns, validated_tickers
        )

        return pd.DataFrame({key: self.data[key] for key in ticker_columns})

    def create_ticker_columns(self, columns, tickers):

        ticker_columns = []
        for tick in tickers:
            for col in columns:
                name = tick + "_" + col
                ticker_columns.append(name)

        return ticker_columns

    def get_tickers(self, ticker_columns):

        return [i.split("_")[0] for i in ticker_columns]
