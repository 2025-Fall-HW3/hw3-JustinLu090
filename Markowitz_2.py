"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:

    def __init__(self, price, exclude, lookback=120, top_n=3, market_timing_window=200):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.top_n = top_n
        self.market_timing_window = market_timing_window
        
        self.spy_sma = Bdf['SPY'].rolling(window=market_timing_window).mean()

    def calculate_weights(self):
        assets = self.price.columns[self.price.columns != self.exclude]

        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        start_index = max(self.lookback, self.market_timing_window) + 1 
        
        for i in range(start_index, len(self.price)):
            current_date = self.price.index[i]
            
            current_spy_price = self.price['SPY'].iloc[i - 1] 
      
            current_spy_sma = self.spy_sma.loc[self.price.index[i - 1]] 

            if current_spy_price < current_spy_sma:
                weights = pd.Series(0.0, index=assets)
            else:
                
                price_window = self.price[assets].iloc[i - self.lookback - 1 : i]
                returns_window = self.returns[assets].iloc[i - self.lookback : i]
                
                price_start = price_window.iloc[0]
                price_end = price_window.iloc[-1]
                momentum = (price_end / price_start) - 1
                top_performers = momentum.sort_values(ascending=False).index[:self.top_n]
                
                volatility = returns_window[top_performers].std()
                inverse_volatility = 1.0 / volatility.replace(0, np.inf)
                
                weights = pd.Series(0.0, index=assets)

                if inverse_volatility.sum() > 0:
                    normalized_weights = inverse_volatility / inverse_volatility.sum()
                    weights.loc[top_performers] = normalized_weights.values
                else:
                    equal_weight = 1.0 / self.top_n
                    weights.loc[top_performers] = equal_weight
            
            self.portfolio_weights.loc[current_date, assets] = weights.values
        
        self.portfolio_weights[self.exclude] = 0
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)