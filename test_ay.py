#%%
import datetime
import pandas as pd
from src.algorithms.backtest import BackTest
from src.algorithms.strategy import CvarMretOpt,MeanSemidevOpt,EqualyWeighted,MeanVariance
from src.datasource.yahoodata import YahooDataSource


tickers = ['MSFT','MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO']

column_name = 'Close'
interval = '1d'

# %%

start_date = datetime.datetime(2020,1,1)
end_date = datetime.datetime(2024,1,1)

# %%
 
main_data = YahooDataSource(tickers,start_date,end_date,columns=[column_name],interval=interval)

data = main_data.get_data()
data
#%%

from src.algorithms.strategy import MeanSemidevOpt,CvarMretOpt
mean_semideviation = CvarMretOpt(1.2)
weights = mean_semideviation.run_startegy(data)
print(weights)

#%%
from src.algorithms.strategy import MeanVariance
returns = data.pct_change().dropna()
returns = pd.DataFrame(returns)
returns
#%%
x=MeanVariance(returns=returns)
x
# %%
