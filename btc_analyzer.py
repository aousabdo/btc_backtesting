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
    significant_drawdowns: List[Dict]

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
        Analyze a trading strategy's performance
        
        Parameters:
        -----------
        portfolio : pd.DataFrame
            Portfolio data with columns: investment, btc_holdings, total_invested, portfolio_value
        price_data : pd.DataFrame
            Price data with daily returns
        """
        # Basic metrics
        total_invested = float(portfolio['total_invested'].iloc[-1])
        final_value = float(portfolio['portfolio_value'].iloc[-1])
        btc_held = float(portfolio['btc_holdings'].iloc[-1])
        roi = ((final_value - total_invested) / total_invested) * 100
        
        # Calculate max drawdown
        rolling_max = portfolio['portfolio_value'].cummax()
        drawdown = (portfolio['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min() * 100)
        
        # Calculate volatility
        daily_returns = portfolio['portfolio_value'].pct_change()
        volatility = float(daily_returns.std() * np.sqrt(252) * 100)  # Annualized
        
        # Calculate Sharpe Ratio
        excess_returns = daily_returns - 0.04/252  # Assuming 4% risk-free rate
        sharpe_ratio = float(np.sqrt(252) * excess_returns.mean() / excess_returns.std())
        
        # Calculate Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = float(np.sqrt(252) * daily_returns.mean() / downside_returns.std())
        
        # Calculate Win Rate
        win_rate = float(len(daily_returns[daily_returns > 0]) / len(daily_returns) * 100)
        
        # Calculate Profit Factor
        gains = daily_returns[daily_returns > 0].sum()
        losses = abs(daily_returns[daily_returns < 0].sum())
        if losses == 0:
            if gains > 0:
                profit_factor = 100.0  # Cap it at 100 instead of infinity
            else:
                profit_factor = 0.0  # If no gains and no losses
        else:
            profit_factor = float(gains / losses)
        
        # Calculate Max Consecutive Losses
        streaks = (daily_returns < 0).astype(int).groupby(
            (daily_returns < 0).astype(int).diff().ne(0).cumsum()
        ).sum()
        max_consecutive_losses = int(streaks[streaks > 0].max() if len(streaks[streaks > 0]) > 0 else 0)
        
        # Calculate yearly returns
        yearly_returns = {}
        for year in portfolio.index.year.unique():
            year_data = portfolio[portfolio.index.year == year]
            if len(year_data) > 0:
                start_value = year_data['portfolio_value'].iloc[0]
                end_value = year_data['portfolio_value'].iloc[-1]
                if start_value > 0:
                    yearly_returns[year] = float(((end_value - start_value) / start_value) * 100)
                else:
                    yearly_returns[year] = 0.0
        
        # Calculate significant drawdown periods
        significant_drawdowns = self._find_significant_drawdowns(portfolio, threshold=-0.1)
        
        return StrategyAnalysis(
            total_invested=total_invested,
            final_value=final_value,
            btc_held=btc_held,
            roi=roi,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_consecutive_losses=max_consecutive_losses,
            yearly_returns=yearly_returns,
            significant_drawdowns=significant_drawdowns
        )
    
    def compare_strategies(self, portfolios: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compare multiple strategies"""
        comparison_data = {}
        
        for name, portfolio in portfolios.items():
            total_invested = portfolio['total_invested'].iloc[-1]
            final_value = portfolio['portfolio_value'].iloc[-1]
            btc_held = portfolio['btc_holdings'].iloc[-1]
            roi = ((final_value - total_invested) / total_invested) * 100
            
            # Calculate max drawdown
            rolling_max = portfolio['portfolio_value'].cummax()
            drawdown = (portfolio['portfolio_value'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Calculate volatility
            daily_returns = portfolio['portfolio_value'].pct_change()
            volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Calculate Sharpe Ratio
            excess_returns = daily_returns - 0.04/252  # Assuming 4% risk-free rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # Calculate Sortino Ratio
            downside_returns = daily_returns[daily_returns < 0]
            sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_returns.std()
            
            # Calculate Win Rate
            win_rate = len(daily_returns[daily_returns > 0]) / len(daily_returns) * 100
            
            # Calculate Profit Factor
            gains = daily_returns[daily_returns > 0].sum()
            losses = abs(daily_returns[daily_returns < 0].sum())
            if losses == 0:
                if gains > 0:
                    profit_factor = 100.0  # Cap it at 100 instead of infinity
                else:
                    profit_factor = 0.0  # If no gains and no losses
            else:
                profit_factor = float(gains / losses)
            
            comparison_data[name] = {
                'Total Invested': total_invested,
                'Final Value': final_value,
                'BTC Held': btc_held,
                'ROI (%)': roi,
                'Max Drawdown (%)': max_drawdown,
                'Volatility (%)': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Win Rate (%)': win_rate,
                'Profit Factor': profit_factor
            }
        
        return pd.DataFrame(comparison_data)
    
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
    
    def _find_significant_drawdowns(self, portfolio: pd.DataFrame, threshold: float = -0.1) -> List[Dict]:
        """Find periods of significant drawdowns"""
        rolling_max = portfolio['portfolio_value'].cummax()
        drawdown = (portfolio['portfolio_value'] - rolling_max) / rolling_max
        
        # Find drawdown periods
        is_drawdown = drawdown < threshold
        shifted_drawdown = is_drawdown.shift(1)
        shifted_drawdown = shifted_drawdown.convert_dtypes()
        shifted_drawdown = shifted_drawdown.fillna(False)
        
        drawdown_starts = is_drawdown & ~shifted_drawdown
        drawdown_ends = (~is_drawdown & shifted_drawdown) | (is_drawdown & (drawdown.index == drawdown.index[-1]))
        
        significant_drawdowns = []
        current_start = None
        
        for date in drawdown.index:
            # Start of a drawdown period
            if drawdown_starts[date]:
                current_start = date
            # End of a drawdown period
            elif drawdown_ends[date] and current_start is not None:
                period_drawdown = drawdown[current_start:date]
                max_drawdown = float(period_drawdown.min() * 100)
                duration = (date - current_start).days
                
                significant_drawdowns.append({
                    'start_date': current_start.strftime('%Y-%m-%d'),
                    'end_date': date.strftime('%Y-%m-%d'),
                    'max_drawdown': max_drawdown,
                    'duration': duration
                })
                current_start = None
        
        # Handle ongoing drawdown at the end of the data
        if current_start is not None:
            period_drawdown = drawdown[current_start:]
            max_drawdown = float(period_drawdown.min() * 100)
            duration = (drawdown.index[-1] - current_start).days
            
            significant_drawdowns.append({
                'start_date': current_start.strftime('%Y-%m-%d'),
                'end_date': drawdown.index[-1].strftime('%Y-%m-%d'),
                'max_drawdown': max_drawdown,
                'duration': duration
            })
        
        # Sort by drawdown magnitude
        significant_drawdowns.sort(key=lambda x: x['max_drawdown'])
        
        return significant_drawdowns
