
# %%

from src.datasource.yahoodata import YahooDataSource
import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# %%
tickers = ['MSFT','MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO']

column_name = 'Close'
interval = '1mo'

# %%

start_date = datetime.datetime(2010,1,1)
end_date = datetime.datetime(2023,1,1)

# %%
 
main_data = YahooDataSource(start_date, end_date, tickers, columns=[column_name], interval=interval).get_data_by_column_tickers(columns=[column_name], tickers=tickers)



main_data
# %%

# Calculate the returns
returns = main_data.pct_change().dropna()
returns = pd.DataFrame(returns)
returns


# %%


# %%


# %%
class MeanVariance:
    
    def __init__(self, returns):
        self.returns = returns
        self.mean_returns = self.calculate_mean()
        self.cov_matrix = self.calculate_covariance_matrix()

    def calculate_mean(self) -> pd.Series:
        mean_returns = self.returns.mean()
        return mean_returns

    def calculate_covariance_matrix(self) -> pd.DataFrame:
        covariance_matrix = self.returns.cov()
        return covariance_matrix

    def portfolio_variance(self, weights):
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))
    
    def return_min_variance_portfolio(self, constraint=None, allow_shorting=False):
        num_assets = len(self.mean_returns)
        initial_weights = np.ones(num_assets) / num_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if allow_shorting:
            bounds = tuple((float('-inf'), float('inf')) for _ in range(num_assets))  
        else:
            bounds = tuple((0, float('inf')) for _ in range(num_assets))  

        if constraint:
            constraints.append(constraint)

        result = minimize(self.portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return dict(zip(self.returns.columns, result.x))
        else:
            raise ValueError("Optimization failed: " + result.message)

    def get_max_return(self) -> float:
        return self.mean_returns.max()

    def minimize_func(self, weights):
        return np.matmul(np.matmul(np.transpose(weights), self.cov_matrix), weights)
    
    def get_optimal_weights(self, target_return=None, constraint=None, allow_shorting=False):
        num_assets = len(self.mean_returns)

        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

        # If a target return is specified, add it as a constraint
        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda weights: np.dot(weights, self.mean_returns) - target_return})

        # If an additional constraint is provided, add it
        if constraint:
            constraints.append(constraint)

        # Set bounds based on whether short selling is allowed
        if allow_shorting:
            bounds = tuple((float('-inf'), float('inf')) for _ in range(num_assets))  
        else:
            bounds = tuple((0, float('inf')) for _ in range(num_assets))  

        # Initial equal distribution
        initial_weights = np.ones(num_assets) / num_assets

        # Optimization
        result = minimize(self.minimize_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        # Check if the optimization was successful
        if result.success:
            return dict(zip(self.returns.columns, result.x))
        else:
            raise ValueError("Optimization failed: " + result.message)
        
# %%

opti = MeanVariance(returns)

# Calculate the maximum achievable return
max_target_return = opti.get_max_return()
print("Maximum Achievable Target Return:")
print(max_target_return)

#Calculate Min Var portfolio weights
min_var_weights = opti.return_min_variance_portfolio()
print("\nMinimum-Variance Portfolio default")
print(min_var_weights)

# Without target return and without short selling
optimal_weights_no_target_no_short = opti.get_optimal_weights(allow_shorting=False)
print("\nOptimal Portfolio Weights (No Target Return, No Short Selling):")
print(optimal_weights_no_target_no_short)

# With a valid target return 
target_return = 0.019  
optimal_weights_target_no_short = opti.get_optimal_weights(target_return=target_return, allow_shorting=False)
print("\nOptimal Portfolio Weights (Target Return, No Short Selling):")
print(optimal_weights_target_no_short)

# With target return and with short selling
target_return = 0.030
optimal_weights_target_short = opti.get_optimal_weights(target_return=target_return, allow_shorting=True)
print("\nOptimal Portfolio Weights (Target Return, Short Selling Allowed):")
print(optimal_weights_target_short)

    
# %%

# for i in range(0,120,5):

#     sub_data = main_data.iloc[i:i+5]

#     mean = opti.calculate_mean(sub_data)

#     covariance = opti.calculate_covariance_matrix(sub_data)

#     optimal_weights = opti.get_optimal_weights(mean,covariance,allow_shorting=True)

# %%
# Initialize variables
initial_wealth = 1000000
results = pd.DataFrame(columns=['Date', 'Wealth', 'Weights', 'Shares'])

# Function to calculate shares
def calculate_shares(weights, wealth, prices):
    return {ticker: int((weights[ticker] * wealth) / prices[ticker]) for ticker in weights}

# Loop through each row
for i in range(12, len(main_data)-1):
    train_data = main_data.iloc[i-12:i]
    curr_data = main_data.iloc[i]
    next_data = main_data.iloc[i+1]
    returns = train_data.pct_change().dropna()
    
    mv = MeanVariance(returns)
    weights = mv.get_optimal_weights()
    
    curr_prices = curr_data
    next_prices=next_data
    shares = calculate_shares(weights, initial_wealth, curr_prices)
    portfolio_value = sum(shares[ticker] * next_prices[ticker] for ticker in shares)
    
    results = pd.concat([results, pd.DataFrame({
        'Date': [curr_data.name],
        'Wealth': [portfolio_value],
        'Weights': [weights],
        'Shares': [shares]
    })], ignore_index=True)
    
    initial_wealth = portfolio_value

# Display the results
print(results)
# %%

# initial_wealth = 1000000
# results = pd.DataFrame(columns=['Date', 'Wealth', 'Weights', 'Shares'])


# def calculate_shares(weights, wealth, prices):
#     return {ticker: int((weights[ticker] * wealth) / prices[ticker]) for ticker in weights}


# for curr_date in pd.date_range(start='2011-01-01', end='2023-02-01', freq='MS'):
    
#     start_train = curr_date - pd.DateOffset(months=12)
#     end_train = curr_date - pd.DateOffset(days=1)
    

#     train_data = main_data.loc[start_train:end_train]
    
    
#     returns = train_data.pct_change().dropna()
    
    
#     mv = MeanVariance(returns)
    

#     weights = mv.return_min_variance_portfolio()
    
    
#     curr_prices = main_data.loc[curr_date]
    
   
#     shares = calculate_shares(weights, initial_wealth, curr_prices)
    
   
#     portfolio_value = sum(shares[ticker] * curr_prices[ticker] for ticker in shares)
    
   
#     initial_wealth = portfolio_value
    
 
#     results = results.append({
#         'Date': curr_date,
#         'Wealth': initial_wealth,
#         'Weights': weights,
#         'Shares': shares
#     }, ignore_index=True)


# print(results)