import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import YearLocator, DateFormatter
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.dates as mdates

class BacktestVisualizer:
    def __init__(self, style: str = 'default'):
        """
        Initialize visualizer with specific style
        
        Args:
            style (str): matplotlib style to use
        """
        plt.style.use('default')
        sns.set_theme()
        sns.set_palette("husl")
        self.mdates = mdates
        plt.style.use('bmh')  # Set default style
        
    def plot_portfolio_values(self, portfolios: Dict[str, pd.DataFrame], save_path: str = None):
        """Plot portfolio values over time"""
        plt.figure(figsize=(12, 6))
        
        for name, portfolio in portfolios.items():
            # Print available columns for debugging
            print(f"Available columns in {name} portfolio:", portfolio.columns.tolist())
            plt.plot(portfolio.index, portfolio['portfolio_value'], label=name)
        
        plt.title('Portfolio Values Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale('log')  # Use log scale
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_btc_holdings(self, portfolios: Dict[str, pd.DataFrame], save_path: str = None):
        """Plot BTC holdings over time"""
        plt.figure(figsize=(12, 6))
        
        for name, portfolio in portfolios.items():
            # Print available columns for debugging
            print(f"Available columns in {name} portfolio:", portfolio.columns.tolist())
            
            # Try different possible column names
            btc_col = None
            possible_names = ['btc_balance', 'btc_amount', 'btc_holdings', 'btc']
            for col in possible_names:
                if col in portfolio.columns:
                    btc_col = col
                    break
            
            if btc_col is None:
                print(f"Warning: No BTC holdings column found in {name} portfolio")
                continue
            
            plt.plot(portfolio.index, portfolio[btc_col], label=name)
        
        plt.title('BTC Holdings Over Time')
        plt.xlabel('Date')
        plt.ylabel('BTC Amount')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale('log')  # Use log scale
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_investment_comparison(self, portfolios: Dict[str, pd.DataFrame], 
                                 save_path: str = None) -> None:
        """
        Plot cumulative investments over time
        """
        plt.figure(figsize=(15, 8))
        
        for strategy, portfolio in portfolios.items():
            plt.plot(portfolio.index, portfolio['total_invested'], 
                    label=strategy, linewidth=2, alpha=0.7)
        
        plt.title('Cumulative Investment Over Time', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Invested ($)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_drawdowns(self, portfolios: Dict[str, pd.DataFrame], save_path: str = None):
        """Plot drawdowns over time"""
        plt.figure(figsize=(12, 6))
        
        for name, portfolio in portfolios.items():
            # Print available columns for debugging
            print(f"Available columns in {name} portfolio:", portfolio.columns.tolist())
            # Calculate drawdown
            rolling_max = portfolio['portfolio_value'].expanding().max()
            drawdown = (portfolio['portfolio_value'] - rolling_max) / rolling_max * 100
            plt.plot(portfolio.index, drawdown, label=name)
        
        plt.title('Portfolio Drawdowns Over Time')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def create_interactive_dashboard(self, portfolios: Dict[str, pd.DataFrame], 
                                  price_data: pd.DataFrame) -> None:
        """
        Create an interactive Plotly dashboard
        """
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Professional color scheme
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Portfolio Values Over Time',
                'BTC Holdings Comparison',
                'Bitcoin Price Movement'
            ),
            vertical_spacing=0.12,
            row_heights=[0.4, 0.3, 0.3]
        )

        # Add portfolio values with improved styling
        for i, (strategy, portfolio) in enumerate(portfolios.items()):
            fig.add_trace(
                go.Scatter(
                    x=portfolio.index,
                    y=portfolio['portfolio_value'],
                    name=f'{strategy}',
                    line=dict(width=2, color=colors[i]),
                    hovertemplate="<b>%{x}</b><br>" +
                                "Value: $%{y:,.2f}<br>" +
                                f"Strategy: {strategy}<extra></extra>"
                ),
                row=1, col=1
            )

        # Add BTC holdings with improved styling
        for i, (strategy, portfolio) in enumerate(portfolios.items()):
            fig.add_trace(
                go.Scatter(
                    x=portfolio.index,
                    y=portfolio['btc_balance'],
                    name=f'{strategy} BTC',
                    line=dict(width=2, color=colors[i], dash='dot'),
                    hovertemplate="<b>%{x}</b><br>" +
                                "BTC: %{y:.4f}<br>" +
                                f"Strategy: {strategy}<extra></extra>"
                ),
                row=2, col=1
            )

        # Add BTC price with improved styling
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['price'],
                name='BTC Price',
                line=dict(width=2, color='#9467bd'),
                fill='tozeroy',
                fillcolor='rgba(148,103,189,0.1)',
                hovertemplate="<b>%{x}</b><br>" +
                            "Price: $%{y:,.2f}<extra></extra>"
            ),
            row=3, col=1
        )

        # Update layout with improved styling
        fig.update_layout(
            height=1200,
            title=dict(
                text="Bitcoin Investment Strategy Analysis Dashboard",
                font=dict(size=24),
                y=0.95
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified'
        )

        # Update axes with improved styling
        for i in range(1, 4):
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False,
                row=i, col=1
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False,
                row=i, col=1
            )

        # Add custom axis labels and formatting
        fig.update_yaxes(
            title_text="Portfolio Value ($)",
            tickprefix="$",
            tickformat=",",
            title_font=dict(size=14),
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="BTC Holdings",
            tickformat=".4f",
            title_font=dict(size=14),
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="BTC Price ($)",
            tickprefix="$",
            tickformat=",",
            title_font=dict(size=14),
            row=3, col=1
        )

        # Add range selector for time periods
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            row=3, col=1
        )

        fig.show()

    def generate_performance_heatmap(self, yearly_returns: Dict[str, Dict[int, float]], 
                                   save_path: str = None) -> None:
        """
        Generate heatmap of yearly returns for each strategy
        """
        # Convert yearly returns to DataFrame
        data = pd.DataFrame(yearly_returns).round(2)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(data, annot=True, cmap='RdYlGn', center=0,
                   fmt='.2f', cbar_kws={'label': 'Return (%)'})
        
        plt.title('Yearly Returns by Strategy (%)', fontsize=14, pad=20)
        plt.ylabel('Year', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_investment_amounts(self, portfolios: Dict[str, pd.DataFrame], save_path: str = None):
        """Plot investment amounts over time"""
        plt.figure(figsize=(12, 6))
        
        # Set up colors for each strategy
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for (name, portfolio), color in zip(portfolios.items(), colors):
            # Calculate cumulative investment for better visualization
            cumulative_investment = portfolio['total_invested'].rolling(window=30, min_periods=1).mean()
            plt.plot(portfolio.index, cumulative_investment, label=name, color=color, linewidth=2)
        
        plt.title('Total Investment Over Time (30-day Moving Average)', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Investment ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.yscale('log')  # Use log scale
        
        # Format y-axis to show dollar amounts
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_locator(YearLocator())
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
        
        # Add light background grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

# Example usage:
if __name__ == "__main__":
    # This would be imported from your backtest results
    # Example:
    # from btc_backtest_core import BacktestCore
    # backtest = BacktestCore('bitcoin_price_data.csv')
    # portfolios = {
    #     'DCA': backtest.run_dca_strategy(),
    #     'Aous': backtest.run_aous_strategy(),
    #     'Lump Sum': backtest.run_lump_sum_strategy(total_investment=100000)
    # }
    
    visualizer = BacktestVisualizer()
    # visualizer.plot_portfolio_values(portfolios)
    # visualizer.create_interactive_dashboard(portfolios, price_data)