import pandas as pd
import numpy as np
from typing import Dict, List, Union
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class StrategyAnalysis:
    """Container for strategy analysis results"""
    total_invested: float
    final_value: float
    btc_held: float
    roi: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    max_consecutive_losses: int
    yearly_returns: Dict[int, float]
    monthly_returns: Dict[str, float]
    drawdown_periods: List[Dict]

class BacktestAnalyzer:
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize analyzer with risk-free rate
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 4%)
        """
        self.risk_free_rate = risk_free_rate
        
    def analyze_strategy(self, portfolio: pd.DataFrame, price_data: pd.DataFrame) -> StrategyAnalysis:
        """
        Perform comprehensive analysis of a strategy
        
        Args:
            portfolio: Portfolio DataFrame with daily values
            price_data: Price DataFrame with daily prices
        
        Returns:
            StrategyAnalysis object with computed metrics
        """
        # Basic metrics
        total_invested = float(portfolio['total_invested'].iloc[-1])
        final_value = float(portfolio['portfolio_value'].iloc[-1])
        btc_held = float(portfolio['btc_balance'].iloc[-1])
        roi = ((final_value - total_invested) / total_invested) * 100 if total_invested > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(portfolio)
        volatility = self._calculate_volatility(portfolio)
        sharpe = self._calculate_sharpe_ratio(portfolio)
        sortino = self._calculate_sortino_ratio(portfolio)
        
        # Trading metrics
        win_rate = self._calculate_win_rate(portfolio)
        profit_factor = self._calculate_profit_factor(portfolio)
        max_consecutive_losses = self._calculate_max_consecutive_losses(portfolio)
        
        # Time-based analysis
        yearly_returns = self._calculate_yearly_returns(portfolio)
        monthly_returns = self._calculate_monthly_returns(portfolio)
        drawdown_periods = self._analyze_drawdown_periods(portfolio)
        
        return StrategyAnalysis(
            total_invested=total_invested,
            final_value=final_value,
            btc_held=btc_held,
            roi=roi,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_consecutive_losses=max_consecutive_losses,
            yearly_returns=yearly_returns,
            monthly_returns=monthly_returns,
            drawdown_periods=drawdown_periods
        )
    
    def compare_strategies(self, strategies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compare multiple strategies side by side
        
        Args:
            strategies: Dictionary of strategy names and their portfolio DataFrames
        
        Returns:
            DataFrame with comparison metrics
        """
        comparison = {}
        
        for name, portfolio in strategies.items():
            analysis = self.analyze_strategy(portfolio, None)
            comparison[name] = {
                'Total Invested': analysis.total_invested,
                'Final Value': analysis.final_value,
                'BTC Held': analysis.btc_held,
                'ROI (%)': analysis.roi,
                'Max Drawdown (%)': analysis.max_drawdown,
                'Volatility (%)': analysis.volatility,
                'Sharpe Ratio': analysis.sharpe_ratio,
                'Sortino Ratio': analysis.sortino_ratio,
                'Win Rate (%)': analysis.win_rate,
                'Profit Factor': analysis.profit_factor
            }
        
        return pd.DataFrame(comparison).round(2)
    
    def generate_report(self, strategy_name: str, analysis: StrategyAnalysis) -> str:
        """Generate detailed text report for a strategy"""
        report = f"""
Strategy Analysis Report: {strategy_name}
{'=' * 50}

Investment Summary:
- Total Invested: ${analysis.total_invested:,.2f}
- Final Value: ${analysis.final_value:,.2f}
- BTC Held: {analysis.btc_held:.4f} BTC
- Return on Investment: {analysis.roi:.2f}%

Risk Metrics:
- Maximum Drawdown: {analysis.max_drawdown:.2f}%
- Annualized Volatility: {analysis.volatility:.2f}%
- Sharpe Ratio: {analysis.sharpe_ratio:.2f}
- Sortino Ratio: {analysis.sortino_ratio:.2f}

Trading Performance:
- Win Rate: {analysis.win_rate:.2f}%
- Profit Factor: {analysis.profit_factor:.2f}
- Max Consecutive Losses: {analysis.max_consecutive_losses}

Yearly Returns:"""
        
        for year, ret in sorted(analysis.yearly_returns.items()):
            report += f"\n- {year}: {ret:.2f}%"
        
        report += "\n\nSignificant Drawdown Periods:"
        for dd in analysis.drawdown_periods:
            report += f"\n- {dd['start']} to {dd['end']}: {dd['depth']:.2f}% ({dd['duration']} days)"
        
        return report
    
    def export_results(self, analysis: StrategyAnalysis, filepath: str) -> None:
        """Export analysis results to JSON file"""
        analysis_dict = {
            'investment_metrics': {
                'total_invested': float(analysis.total_invested),
                'final_value': float(analysis.final_value),
                'btc_held': float(analysis.btc_held),
                'roi': float(analysis.roi)
            },
            'risk_metrics': {
                'max_drawdown': float(analysis.max_drawdown),
                'volatility': float(analysis.volatility),
                'sharpe_ratio': float(analysis.sharpe_ratio),
                'sortino_ratio': float(analysis.sortino_ratio)
            },
            'trading_metrics': {
                'win_rate': float(analysis.win_rate),
                'profit_factor': float(analysis.profit_factor),
                'max_consecutive_losses': int(analysis.max_consecutive_losses)
            },
            'yearly_returns': {str(k): float(v) for k, v in analysis.yearly_returns.items()},
            'monthly_returns': {str(k): float(v) for k, v in analysis.monthly_returns.items()},
            'drawdown_periods': [
                {
                    'start': str(dd['start']),
                    'end': str(dd['end']),
                    'depth': float(dd['depth']),
                    'duration': int(dd['duration'])
                }
                for dd in analysis.drawdown_periods
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_dict, f, indent=4)
    
    def _calculate_max_drawdown(self, portfolio: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage"""
        rolling_max = portfolio['portfolio_value'].cummax()
        drawdown = (portfolio['portfolio_value'] - rolling_max) / rolling_max
        return float(drawdown.min() * 100)
    
    def _calculate_volatility(self, portfolio: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        daily_returns = portfolio['portfolio_value'].pct_change().dropna()
        return float(daily_returns.std() * np.sqrt(252) * 100)
    
    def _calculate_sharpe_ratio(self, portfolio: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        daily_returns = portfolio['portfolio_value'].pct_change().dropna()
        excess_returns = daily_returns - (self.risk_free_rate / 252)
        if len(excess_returns) == 0 or excess_returns.std() == 0:
            return 0.0
        return float(np.sqrt(252) * (excess_returns.mean() / excess_returns.std()))
    
    def _calculate_sortino_ratio(self, portfolio: pd.DataFrame) -> float:
        """Calculate Sortino ratio"""
        daily_returns = portfolio['portfolio_value'].pct_change().dropna()
        excess_returns = daily_returns - (self.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf')
            
        return float(np.sqrt(252) * (excess_returns.mean() / downside_returns.std()))
    
    def _calculate_win_rate(self, portfolio: pd.DataFrame) -> float:
        """Calculate percentage of winning days"""
        daily_returns = portfolio['portfolio_value'].pct_change().dropna()
        winning_days = (daily_returns > 0).sum()
        return float((winning_days / len(daily_returns)) * 100)
    
    def _calculate_profit_factor(self, portfolio: pd.DataFrame) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        daily_returns = portfolio['portfolio_value'].pct_change().dropna()
        gross_profits = daily_returns[daily_returns > 0].sum()
        gross_losses = abs(daily_returns[daily_returns < 0].sum())
        
        if gross_losses == 0:
            return float('inf')
            
        return float(gross_profits / gross_losses)
    
    def _calculate_max_consecutive_losses(self, portfolio: pd.DataFrame) -> int:
        """Calculate maximum consecutive losing days"""
        daily_returns = portfolio['portfolio_value'].pct_change().dropna()
        losses = (daily_returns < 0).astype(int)
        
        max_consecutive = 0
        current_consecutive = 0
        
        for loss in losses:
            if loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return int(max_consecutive)
    
    def _calculate_yearly_returns(self, portfolio: pd.DataFrame) -> Dict[int, float]:
        """Calculate returns for each year"""
        yearly_returns = {}
        
        for year in portfolio.index.year.unique():
            year_data = portfolio[portfolio.index.year == year]
            start_invested = year_data['total_invested'].iloc[0]
            end_invested = year_data['total_invested'].iloc[-1]
            start_value = year_data['portfolio_value'].iloc[0]
            end_value = year_data['portfolio_value'].iloc[-1]
            
            if start_invested == end_invested:  # No new investments in this year
                if start_value > 0:
                    yearly_returns[int(year)] = float(((end_value - start_value) / start_value) * 100)
                else:
                    yearly_returns[int(year)] = 0.0
            else:  # New investments were made
                investment_return = ((end_value - end_invested) / end_invested) * 100
                yearly_returns[int(year)] = float(investment_return)
            
        return yearly_returns
    
    def _calculate_monthly_returns(self, portfolio: pd.DataFrame) -> Dict[str, float]:
        """Calculate returns for each month"""
        monthly_returns = {}
        
        for year in portfolio.index.year.unique():
            for month in portfolio[portfolio.index.year == year].index.month.unique():
                month_data = portfolio[
                    (portfolio.index.year == year) & 
                    (portfolio.index.month == month)
                ]
                
                start_invested = month_data['total_invested'].iloc[0]
                end_invested = month_data['total_invested'].iloc[-1]
                start_value = month_data['portfolio_value'].iloc[0]
                end_value = month_data['portfolio_value'].iloc[-1]
                
                month_key = f"{year}-{month:02d}"
                
                if start_invested == end_invested:  # No new investments in this month
                    if start_value > 0:
                        monthly_returns[month_key] = float(((end_value - start_value) / start_value) * 100)
                    else:
                        monthly_returns[month_key] = 0.0
                else:  # New investments were made
                    investment_return = ((end_value - end_invested) / end_invested) * 100
                    monthly_returns[month_key] = float(investment_return)
        
        return monthly_returns
    
    def _analyze_drawdown_periods(self, portfolio: pd.DataFrame) -> List[Dict]:
        """Analyze significant drawdown periods (>10%)"""
        rolling_max = portfolio['portfolio_value'].cummax()
        drawdown = (portfolio['portfolio_value'] - rolling_max) / rolling_max * 100
        
        significant_drawdowns = []
        in_drawdown = False
        start_date = None
        threshold = -10  # Only track drawdowns greater than 10%
        
        dates = portfolio.index
        for i, date in enumerate(dates):
            if not in_drawdown and drawdown[date] <= threshold:
                in_drawdown = True
                start_date = date
            elif in_drawdown:
                # Check if drawdown has ended or we're at the end of the data
                if drawdown[date] > threshold or i == len(dates) - 1:
                    end_date = dates[i-1] if drawdown[date] > threshold else date
                    duration = (end_date - start_date).days
                    
                    # Find the deepest drawdown in this period
                    period_drawdown = drawdown[start_date:end_date]
                    min_date = period_drawdown.idxmin()
                    depth = drawdown[min_date]
                    
                    significant_drawdowns.append({
                        'start': start_date.strftime('%Y-%m-%d'),
                        'end': end_date.strftime('%Y-%m-%d'),
                        'depth': float(depth),
                        'duration': int(duration)
                    })
                    
                    in_drawdown = False
                    start_date = None
        
        return sorted(significant_drawdowns, key=lambda x: x['depth'])
