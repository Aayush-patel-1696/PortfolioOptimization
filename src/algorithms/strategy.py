#%%
import pandas as pd
import numpy as np
from scipy import optimize
from scipy.optimize import minimize
#%%
"""
Contains the definition of abstract class ScenarioGen
"""
from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    Contains abstract methods for scenario generation
    """

    @abstractmethod
    def get_optimal_allocations(self,*args,**kwargs):
        """
        Get the Optimal weights 
        """

    @abstractmethod
    def run_startegy(self,*args,**kwargs):
        """
        Run strategy between this ind
        """

class ConstrainedBasedStrategy(Strategy):

    def run_startegy(self,price_data,test_size=30*3,strategy_frequency=30):

        weights_dict = {}
        
        min_date = (price_data.index[0])
        max_date = (price_data.index[-1])

        date_range = []
        current_date = min_date + pd.DateOffset(test_size)
        while current_date < max_date:
              date_range.append(current_date)
              current_date  = current_date + pd.DateOffset(strategy_frequency)

        for index,date in enumerate(date_range[:-2]):

            start_date = date - pd.DateOffset(test_size) 
            end_date = date

            if end_date <= max_date:
                pass
            else:
                continue

            filtered_price_data = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
            filtered_rtn_data = filtered_price_data.pct_change()[1:]

            wealth_allocations = self.get_optimal_allocations(filtered_rtn_data.T.iloc[:,1:],1)
            weights_dict[date_range[index+1]] = dict(zip(price_data.columns,wealth_allocations))

        return weights_dict
    




class CvarMretOpt(ConstrainedBasedStrategy):


    def __init__(self,ratio=0.5,risk_level=0.3):


        self.ratio = ratio
        self.risk_level = risk_level
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None
        

    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):
        self.array = returns_data.iloc[:, 1:].to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = investment_amount 
        self.results = self.optimize(self.ratio,self.risk_level)
        return self.results.x[self.num_senarios:self.num_senarios+self.num_assets]
    
    def get_cvar_value(self):
        return self.results.x[-1]

    def optimize(self,ratio,risk_level):

        """Solve the problem of minimizing the function 
                -(1-c) E[Z(x)] + c AVaR[Z(x)]
        """

        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))

        lhs_ineq = np.zeros((self.num_senarios,self.num_senarios+self.num_assets+1))

        for i in range(self.num_senarios):
            lhs_ineq[i,i] = -1  # vk
            lhs_ineq[i,self.num_senarios:self.num_senarios+self.num_assets] = -1*(self.array[:,i]) # Rk  
            lhs_ineq[i,-1] = 1    # n

        rhs_ineq = np.zeros((1,self.num_senarios))

        lhs_eq = np.zeros((1,self.num_senarios+self.num_assets+1))
        lhs_eq[0,self.num_senarios:self.num_senarios+self.num_assets] = 1
        rhs_eq = [self.investment_amount]

        bnd = []
        for i in range(self.num_senarios+self.num_assets+1):
            bnd.append((0,float('inf')))

        bnd[-1] = (float('-inf'),float('inf'))

        obj = np.ones((1,self.num_senarios+self.num_assets+1))*(1/risk_level)*(1/self.num_senarios)*(ratio)
        obj[0,-1] = -1*(ratio)
        obj[0,self.num_senarios:self.num_senarios+self.num_assets] = -1*(1-ratio)*np.array(np.transpose(mean))
        opt = optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,method="highs")

        return opt

        

class MeanSemidevOpt(ConstrainedBasedStrategy):

    def __init__(self,ratio):
       
        self.ratio = ratio 
        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None
        
    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.iloc[:, 1:].to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = investment_amount
        self.results = self.optimize(self.ratio)
        return self.results.x[self.num_senarios:self.num_senarios+self.num_assets]

    def optimize(self,ratio):

        """
            ρ =  Hybrid Risk Measure
            ρ = [Z(x)] = -E[Z(x)] + c σ[Z(x)]

            where σ[Z] = E{ max(0,E[Z] – Z)} is the lower semideviation of the first order.
        """

        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))
        lhs_ineq = np.zeros((self.num_senarios,self.num_senarios+self.num_assets))

        for i in range(self.num_senarios):
            lhs_ineq[i,i] = -1  # vk
            lhs_ineq[i,self.num_senarios:self.num_senarios+self.num_assets] = -1*(self.array[:,i]) + np.transpose(mean)  # Rk

        rhs_ineq = np.zeros((1,self.num_senarios))

        lhs_eq = np.zeros((1,self.num_senarios+self.num_assets))
        lhs_eq[0,self.num_senarios:self.num_senarios+self.num_assets] = 1
        rhs_eq = [self.investment_amount]

        bnd = []
        for i in range(self.num_senarios+self.num_assets):
            bnd.append((0,float('inf')))

        obj = np.ones((1,self.num_senarios+self.num_assets))*(1/self.num_senarios)*ratio
        obj[0,self.num_senarios:self.num_senarios+self.num_assets] = -1*np.array(np.transpose(mean))

        opt = optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,method="highs")
        return opt
        


class EqualyWeighted(ConstrainedBasedStrategy):

    def __init__(self):
    
        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None

    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.iloc[:, 1:].to_numpy()
        self.num_assets = len(self.array[:,0])
        return (np.ones((1,self.num_assets))/self.num_assets)*investment_amount
    
class MeanVariance(ConstrainedBasedStrategy):
    
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
    
    def min_variance(self, allow_shorting=False):
        num_assets = len(self.mean_returns)
        initial_weights = np.ones(num_assets) / num_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if allow_shorting:
            bounds = tuple((float('-inf'), float('inf')) for _ in range(num_assets))  
        else:
            bounds = tuple((0, float('inf')) for _ in range(num_assets))  
        
        result = minimize(self.portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result
    
    def min_variance_allocation(self):
        
        result = self.min_variance()
        if result.success:
            return dict(zip(self.returns.columns, result.x))
        else:
            raise ValueError("Optimization failed: " + result.message)


    def get_max_return(self) -> float:
        return self.mean_returns.max()

    def minimize_func(self, weights):
        return np.matmul(np.matmul(np.transpose(weights), self.cov_matrix), weights)
    
    def optimize(self, target_return=None, allow_shorting=False):
        
        num_assets = len(self.mean_returns)
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda weights: np.dot(weights, self.mean_returns) - target_return})
        if allow_shorting:
            bounds = tuple((float('-inf'), float('inf')) for _ in range(num_assets))  
        else:
            bounds = tuple((0, float('inf')) for _ in range(num_assets))  
        initial_weights = np.ones(num_assets) / num_assets
        result = minimize(self.minimize_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def get_optimal_allocations(self):
        
        result = self.optimize()
        if result.success:
            return dict(zip(self.returns.columns, result.x))
        else:
            raise ValueError("Optimization failed: " + result.message)
# %%
