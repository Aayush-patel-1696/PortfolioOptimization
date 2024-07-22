from .strategy import Strategy
import numpy as np
import pandas as pd

class BackTest:

    def __init__(self,price_data:pd.DataFrame,method:Strategy,test_index:int=10,initial_wealth=1): # To DO: Start_date and End-Date params

        self.prc_data = price_data
        self.test_index = test_index
        self.initial_wealth = initial_wealth
        self.method = method

    def backtest(self):

        test_index = self.test_index
        price_data = self.prc_data
        return_data = self.prc_data.pct_change()
        wealth = self.initial_wealth
        portfolio_wealths = [wealth]

        for i in range(0,len(price_data)-test_index-1):

            temp_return_data = return_data.iloc[1+i:test_index+i]
            wealth_allocations = self.method.get_optimal_allocations(temp_return_data.T.iloc[:,1:],wealth)
            shares_allocations = wealth_allocations/(price_data.iloc[test_index+i].to_numpy())
            wealth = np.sum((price_data.iloc[test_index+i+1].to_numpy())*(shares_allocations))
            portfolio_wealths.append(wealth)

        return portfolio_wealths
    

    






        