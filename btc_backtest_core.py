import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Tuple

class BacktestCore:
    def __init__(self, csv_file: str):
        """
        Initialize backtesting engine with historical price data
        
        Args:
            csv_file (str): Path to CSV file with columns: timestamp, price, date
        """
        self.df = self._load_and_prepare_data(csv_file)
        self.total_days = len(self.df)
    
    def _load_and_prepare_data(self, csv_file: str) -> pd.DataFrame:
        """Load and prepare the price data"""
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Convert date to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Sort by date
        df = df.sort_index()
        
        # Ensure numeric types
        df['price'] = pd.to_numeric(df['price'])
        
        # Calculate daily returns
        df['daily_return'] = df['price'].pct_change()
        
        return df
    
    def run_dca_strategy(self, daily_investment: float = 100) -> Dict[str, pd.Series]:
        """
        Run Dollar Cost Averaging strategy
        
        Args:
            daily_investment (float): Amount to invest daily
        
        Returns:
            Dict containing portfolio metrics
        """
        portfolio = pd.DataFrame(index=self.df.index)
        portfolio['investment'] = daily_investment
        portfolio['btc_bought'] = portfolio['investment'] / self.df['price']
        portfolio['btc_balance'] = portfolio['btc_bought'].cumsum()
        portfolio['portfolio_value'] = portfolio['btc_balance'] * self.df['price']
        portfolio['total_invested'] = portfolio['investment'].cumsum()
        
        return portfolio
    
    def run_aous_strategy(self, daily_investment: float = 100, 
                         dip_investment: float = 1000) -> Dict[str, pd.Series]:
        """
        Run Aous's strategy (DCA + buy dips)
        
        Args:
            daily_investment (float): Base daily investment
            dip_investment (float): Additional investment on down days
        
        Returns:
            Dict containing portfolio metrics
        """
        portfolio = pd.DataFrame(index=self.df.index)
        
        # Base investment + extra on down days
        portfolio['investment'] = daily_investment
        portfolio.loc[self.df['daily_return'] < 0, 'investment'] += dip_investment
        
        portfolio['btc_bought'] = portfolio['investment'] / self.df['price']
        portfolio['btc_balance'] = portfolio['btc_bought'].cumsum()
        portfolio['portfolio_value'] = portfolio['btc_balance'] * self.df['price']
        portfolio['total_invested'] = portfolio['investment'].cumsum()
        
        return portfolio
    
    def run_lump_sum_strategy(self, total_investment: float) -> Dict[str, pd.Series]:
        """
        Run Lump Sum strategy
        
        Args:
            total_investment (float): Total amount to invest at start
        
        Returns:
            Dict containing portfolio metrics
        """
        portfolio = pd.DataFrame(index=self.df.index)
        portfolio['investment'] = 0.0  # Set float dtype
        portfolio['btc_bought'] = 0.0  # Set float dtype
        
        portfolio.iloc[0, portfolio.columns.get_loc('investment')] = float(total_investment)
        portfolio.iloc[0, portfolio.columns.get_loc('btc_bought')] = float(total_investment / self.df['price'].iloc[0])
        
        portfolio['btc_balance'] = portfolio['btc_bought'].cumsum()
        portfolio['portfolio_value'] = portfolio['btc_balance'] * self.df['price']
        portfolio['total_invested'] = portfolio['investment'].cumsum()
        
        return portfolio
    
    def calculate_metrics(self, portfolio: pd.DataFrame) -> Dict[str, Union[float, Dict]]:
        """
        Calculate performance metrics for a strategy
        
        Args:
            portfolio (pd.DataFrame): Portfolio data
        
        Returns:
            Dict containing calculated metrics
        """
        # Basic metrics
        total_invested = portfolio['total_invested'].iloc[-1]
        final_value = portfolio['portfolio_value'].iloc[-1]
        btc_held = portfolio['btc_balance'].iloc[-1]
        roi = ((final_value - total_invested) / total_invested) * 100
        
        # Calculate max drawdown
        rolling_max = portfolio['portfolio_value'].cummax()
        drawdown = (portfolio['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Calculate volatility
        daily_returns = portfolio['portfolio_value'].pct_change()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Calculate yearly returns
        yearly_returns = {}
        for year in portfolio.index.year.unique():
            year_data = portfolio[portfolio.index.year == year]
            start_invested = year_data['total_invested'].iloc[0]
            end_invested = year_data['total_invested'].iloc[-1]
            start_value = year_data['portfolio_value'].iloc[0]
            end_value = year_data['portfolio_value'].iloc[-1]
            
            # Calculate return based on the change in portfolio value relative to total investment
            if start_invested == end_invested:  # No new investments in this year
                if start_value > 0:
                    yearly_returns[year] = ((end_value - start_value) / start_value) * 100
                else:
                    yearly_returns[year] = 0
            else:  # New investments were made
                investment_return = ((end_value - end_invested) / end_invested) * 100
                yearly_returns[year] = investment_return

        return {
            'total_invested': total_invested,
            'final_value': final_value,
            'btc_held': btc_held,
            'roi': roi,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'yearly_returns': yearly_returns
        }
    
    def get_price_metrics(self) -> Dict[str, float]:
        """Get Bitcoin price metrics"""
        return {
            'start_price': self.df['price'].iloc[0],
            'end_price': self.df['price'].iloc[-1],
            'price_return': ((self.df['price'].iloc[-1] / self.df['price'].iloc[0]) - 1) * 100,
            'down_days': len(self.df[self.df['daily_return'] < 0]),
            'total_days': self.total_days
        }

# Usage example:
if __name__ == "__main__":
    # Initialize backtest
    backtest = BacktestCore('bitcoin_price_data.csv')
    
    # Get price metrics
    price_metrics = backtest.get_price_metrics()
    print("\nBitcoin Price Metrics:")
    print(f"Start Price: ${price_metrics['start_price']:,.2f}")
    print(f"End Price: ${price_metrics['end_price']:,.2f}")
    print(f"Return: {price_metrics['price_return']:.2f}%")
    print(f"Down Days: {price_metrics['down_days']} out of {price_metrics['total_days']}")