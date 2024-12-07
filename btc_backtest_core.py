import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Tuple

class BacktestCore:
    def __init__(self, price_data_file: str, price_col: str = 'price', date_col: str = 'date'):
        """
        Initialize backtesting core
        
        Parameters:
        -----------
        price_data_file : str
            Path to CSV file containing price data
        price_col : str
            Name of the price column in the CSV file
        date_col : str
            Name of the date column in the CSV file
        """
        self.df = pd.read_csv(price_data_file)
        
        # Ensure we have the required columns
        if date_col not in self.df.columns:
            raise ValueError(f"Date column '{date_col}' not found in data. Available columns: {list(self.df.columns)}")
        if price_col not in self.df.columns:
            raise ValueError(f"Price column '{price_col}' not found in data. Available columns: {list(self.df.columns)}")
        
        # Convert date and set as index
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df.set_index(date_col, inplace=True)
        
        self.price_col = price_col
        self.df['daily_return'] = self.df[price_col].pct_change()

    def calculate_rsi(self, data: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI for a given price series"""
        # Calculate price changes
        delta = data.diff()
        
        # Create gain and loss series
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        # Calculate average gains and losses
        avg_gains = pd.Series(index=data.index)
        avg_losses = pd.Series(index=data.index)
        
        # First values are just the first gains/losses
        avg_gains.iloc[periods-1] = gain.iloc[:periods].mean()
        avg_losses.iloc[periods-1] = loss.iloc[:periods].mean()
        
        # Calculate subsequent values using the previous averages
        for i in range(periods, len(data)):
            avg_gains.iloc[i] = (avg_gains.iloc[i-1] * (periods-1) + gain.iloc[i]) / periods
            avg_losses.iloc[i] = (avg_losses.iloc[i-1] * (periods-1) + loss.iloc[i]) / periods
        
        # Calculate RS and RSI
        rs = pd.Series(index=data.index)
        rsi = pd.Series(index=data.index)
        
        for i in range(periods, len(data)):
            if avg_losses.iloc[i] == 0:
                rs.iloc[i] = 100
            else:
                rs.iloc[i] = avg_gains.iloc[i] / avg_losses.iloc[i]
            rsi.iloc[i] = 100 - (100 / (1 + rs.iloc[i]))
        
        return rsi

    def calculate_moving_averages(self, data: pd.Series) -> pd.DataFrame:
        """Calculate multiple moving averages"""
        ma_periods = {
            'MA20': 20,
            'MA50': 50,
            'MA200': 200
        }
        
        ma_df = pd.DataFrame(index=data.index)
        for name, period in ma_periods.items():
            ma_df[name] = data.rolling(window=period).mean()
        
        return ma_df

    def run_dca_strategy(self, daily_investment: float = 100) -> pd.DataFrame:
        """Run basic DCA strategy"""
        portfolio = self._initialize_portfolio()
        
        # Track investments
        total_invested = 0
        btc_holdings = 0
        
        for date in portfolio.index:
            current_price = self.df.loc[date, self.price_col]
            
            # Execute investment
            btc_bought = daily_investment / current_price
            total_invested += daily_investment
            btc_holdings += btc_bought
            
            # Update portfolio
            portfolio.loc[date, 'investment'] = daily_investment
            portfolio.loc[date, 'btc_bought'] = btc_bought
            portfolio.loc[date, 'total_invested'] = total_invested
            portfolio.loc[date, 'btc_holdings'] = btc_holdings
            portfolio.loc[date, 'portfolio_value'] = btc_holdings * current_price
        
        return portfolio

    def run_aous_strategy(self, daily_investment: float = 100, 
                         dip_investment: float = 1000,
                         dip_threshold: float = 0.1,
                         holding_period: int = 30) -> pd.DataFrame:
        """
        Run Enhanced DCA with Dip Buying strategy
        
        Parameters:
        -----------
        daily_investment : float
            Base daily investment amount
        dip_investment : float
            Additional investment during dips
        dip_threshold : float
            Price drop threshold to trigger dip buying (e.g., 0.1 for 10%)
        holding_period : int
            Number of days to hold each position
        """
        portfolio = self._initialize_portfolio()
        
        # Track investments
        total_invested = 0
        btc_holdings = 0
        
        # Calculate rolling high for dip detection
        rolling_high = self.df[self.price_col].rolling(window=holding_period).max()
        
        for date in portfolio.index:
            current_price = self.df.loc[date, self.price_col]
            
            # Calculate price drop from recent high
            if date == portfolio.index[0]:
                price_drop = 0
            else:
                recent_high = rolling_high.loc[date]
                price_drop = (recent_high - current_price) / recent_high
            
            # Determine investment amount
            if price_drop >= dip_threshold:
                investment = daily_investment + dip_investment
            else:
                investment = daily_investment
            
            # Execute investment
            btc_bought = investment / current_price
            total_invested += investment
            btc_holdings += btc_bought
            
            # Update portfolio
            portfolio.loc[date, 'investment'] = investment
            portfolio.loc[date, 'btc_bought'] = btc_bought
            portfolio.loc[date, 'total_invested'] = total_invested
            portfolio.loc[date, 'btc_holdings'] = btc_holdings
            portfolio.loc[date, 'portfolio_value'] = btc_holdings * current_price
        
        return portfolio

    def run_lump_sum_strategy(self, total_investment: float) -> pd.DataFrame:
        """Run lump sum investment strategy"""
        portfolio = self._initialize_portfolio()
        
        # Invest everything on day 1
        first_date = portfolio.index[0]
        first_price = self.df.loc[first_date, self.price_col]
        btc_holdings = total_investment / first_price
        
        for date in portfolio.index:
            current_price = self.df.loc[date, self.price_col]
            
            # Update portfolio
            portfolio.loc[date, 'investment'] = total_investment if date == first_date else 0
            portfolio.loc[date, 'btc_bought'] = btc_holdings if date == first_date else 0
            portfolio.loc[date, 'total_invested'] = total_investment
            portfolio.loc[date, 'btc_holdings'] = btc_holdings
            portfolio.loc[date, 'portfolio_value'] = btc_holdings * current_price
        
        return portfolio

    def run_rsi_strategy(self, base_investment: float = 100, 
                        rsi_thresholds: dict = None,
                        rsi_period: int = 14) -> pd.DataFrame:
        """
        RSI-based investment strategy
        
        Parameters:
        -----------
        base_investment : float
            Base daily investment amount
        rsi_thresholds : dict
            Dictionary of RSI thresholds and their corresponding additional investments
            e.g., {30: 2000, 20: 5000} means:
            - Add $2000 when RSI <= 30
            - Add $5000 when RSI <= 20
        rsi_period : int
            Period for RSI calculation
        """
        if rsi_thresholds is None:
            rsi_thresholds = {30: 2000, 20: 5000}
        
        portfolio = self._initialize_portfolio()
        
        # Calculate RSI
        rsi = self.calculate_rsi(self.df[self.price_col], rsi_period)
        
        # Track investments
        total_invested = 0
        btc_holdings = 0
        
        for date in portfolio.index:
            current_price = self.df.loc[date, self.price_col]
            current_rsi = rsi[date]
            
            # Determine investment amount based on RSI
            investment = base_investment
            if not pd.isna(current_rsi):  # Only add extra investment if RSI is valid
                for threshold, extra_amount in sorted(rsi_thresholds.items(), reverse=True):
                    if current_rsi <= threshold:
                        investment += extra_amount
                        break
            
            # Execute investment if price is valid
            if current_price > 0 and not pd.isna(current_price):
                btc_bought = investment / current_price
                total_invested += investment
                btc_holdings += btc_bought
            else:
                btc_bought = 0
            
            # Update portfolio
            portfolio.loc[date, 'investment'] = investment
            portfolio.loc[date, 'btc_bought'] = btc_bought
            portfolio.loc[date, 'total_invested'] = total_invested
            portfolio.loc[date, 'btc_holdings'] = btc_holdings
            portfolio.loc[date, 'portfolio_value'] = btc_holdings * current_price
        
        return portfolio

    def run_ma_momentum_strategy(self, base_investment: float = 100,
                               ma_multipliers: dict = None) -> pd.DataFrame:
        """
        Moving Average Momentum strategy
        """
        if ma_multipliers is None:
            ma_multipliers = {'MA20': 2, 'MA50': 3}
        
        portfolio = self._initialize_portfolio()
        
        # Calculate Moving Averages
        ma_df = self.calculate_moving_averages(self.df[self.price_col])
        
        # Track investments
        total_invested = 0
        btc_holdings = 0
        
        for date in portfolio.index:
            current_price = self.df.loc[date, self.price_col]
            
            # Determine investment multiplier based on MA positions
            multiplier = 1
            for ma_name, mult in sorted(ma_multipliers.items(), key=lambda x: int(x[0][2:])):
                if current_price < ma_df.loc[date, ma_name]:
                    multiplier = mult
                    break
            
            # Calculate investment amount
            investment = base_investment * multiplier
            
            # Execute investment
            btc_bought = investment / current_price
            total_invested += investment
            btc_holdings += btc_bought
            
            # Update portfolio
            portfolio.loc[date, 'investment'] = investment
            portfolio.loc[date, 'btc_bought'] = btc_bought
            portfolio.loc[date, 'total_invested'] = total_invested
            portfolio.loc[date, 'btc_holdings'] = btc_holdings
            portfolio.loc[date, 'portfolio_value'] = btc_holdings * current_price
        
        return portfolio

    def _initialize_portfolio(self) -> pd.DataFrame:
        """Initialize portfolio DataFrame with required columns"""
        portfolio = pd.DataFrame(index=self.df.index)
        portfolio['investment'] = 0.0
        portfolio['btc_bought'] = 0.0
        portfolio['total_invested'] = 0.0
        portfolio['btc_holdings'] = 0.0
        portfolio['portfolio_value'] = 0.0
        return portfolio

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