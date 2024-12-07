import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from itertools import product
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

@dataclass
class SimulationParams:
    """Parameters for simulation"""
    base_investments: List[float]
    dip_investments: List[float]
    dip_thresholds: List[float]  # Percentage drops that trigger dip buying
    holding_periods: List[int]   # Days to hold each position

class ParameterSimulator:
    def __init__(self, backtest_core):
        """
        Initialize parameter simulator with backtest core instance
        
        Args:
            backtest_core: Instance of BacktestCore class
        """
        self.core = backtest_core
        self.results_cache = {}

    def generate_parameter_combinations(self, params: SimulationParams) -> List[Dict]:
        """Generate all possible parameter combinations"""
        combinations = []
        
        for base, dip, threshold, period in product(
            params.base_investments,
            params.dip_investments,
            params.dip_thresholds,
            params.holding_periods
        ):
            combinations.append({
                'base_investment': base,
                'dip_investment': dip,
                'dip_threshold': threshold,
                'holding_period': period
            })
        
        return combinations

    def run_single_simulation(self, params: Dict) -> Dict:
        """Run a single simulation with given parameters"""
        try:
            # Create unique key for caching
            param_key = tuple(sorted(params.items()))
            
            # Check cache
            if param_key in self.results_cache:
                return self.results_cache[param_key]
            
            # Run strategies with current parameters
            dca_portfolio = self.core.run_dca_strategy(params['base_investment'])
            aous_portfolio = self.core.run_aous_strategy(
                params['base_investment'], 
                params['dip_investment']
            )
            lump_sum_portfolio = self.core.run_lump_sum_strategy(
                params['base_investment'] * len(self.core.df)
            )
            
            # Calculate metrics
            dca_metrics = self.core.calculate_metrics(dca_portfolio)
            aous_metrics = self.core.calculate_metrics(aous_portfolio)
            lump_sum_metrics = self.core.calculate_metrics(lump_sum_portfolio)
            
            # Convert datetime index to string for serialization
            dca_portfolio.index = dca_portfolio.index.strftime('%Y-%m-%d')
            aous_portfolio.index = aous_portfolio.index.strftime('%Y-%m-%d')
            lump_sum_portfolio.index = lump_sum_portfolio.index.strftime('%Y-%m-%d')
            
            result = {
                'parameters': params,
                'dca_metrics': dca_metrics,
                'aous_metrics': aous_metrics,
                'lump_sum_metrics': lump_sum_metrics,
                'portfolios': {
                    'dca': dca_portfolio,
                    'aous': aous_portfolio,
                    'lump_sum': lump_sum_portfolio
                }
            }
            
            # Cache result
            self.results_cache[param_key] = result
            
            return result
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            return None

    def run_parallel_simulations(self, params: SimulationParams, 
                               max_workers: int = None) -> List[Dict]:
        """Run simulations in parallel"""
        combinations = self.generate_parameter_combinations(params)
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {
                executor.submit(self.run_single_simulation, params): params 
                for params in combinations
            }
            
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f'Simulation failed for params {params}: {str(e)}')
        
        return results

    def analyze_simulation_results(self, results: List[Dict]) -> pd.DataFrame:
        """Analyze simulation results and return as DataFrame"""
        analysis = []
        
        for result in results:
            if result is None:
                continue
                
            params = result['parameters']
            
            try:
                analysis.append({
                    'base_investment': params['base_investment'],
                    'dip_investment': params['dip_investment'],
                    'dip_threshold': params['dip_threshold'],
                    'holding_period': params['holding_period'],
                    'dca_roi': result['dca_metrics']['roi'],
                    'aous_roi': result['aous_metrics']['roi'],
                    'lump_sum_roi': result['lump_sum_metrics']['roi'],
                    'dca_max_drawdown': result['dca_metrics']['max_drawdown'],
                    'aous_max_drawdown': result['aous_metrics']['max_drawdown'],
                    'lump_sum_max_drawdown': result['lump_sum_metrics']['max_drawdown'],
                    'dca_sharpe': result['dca_metrics'].get('sharpe_ratio', 0),
                    'aous_sharpe': result['aous_metrics'].get('sharpe_ratio', 0),
                    'lump_sum_sharpe': result['lump_sum_metrics'].get('sharpe_ratio', 0),
                    'dca_btc_held': result['dca_metrics']['btc_held'],
                    'aous_btc_held': result['aous_metrics']['btc_held'],
                    'lump_sum_btc_held': result['lump_sum_metrics']['btc_held']
                })
            except Exception as e:
                print(f"Error analyzing result: {str(e)}")
                continue
        
        if not analysis:
            return pd.DataFrame()
            
        return pd.DataFrame(analysis)

    def find_optimal_parameters(self, results_df: pd.DataFrame, 
                              optimization_target: str = 'roi',
                              risk_adjusted: bool = True) -> Dict:
        """
        Find optimal parameters based on specified target
        """
        if results_df.empty:
            return {}
            
        if optimization_target == 'roi':
            target_cols = ['dca_roi', 'aous_roi', 'lump_sum_roi']
        elif optimization_target == 'drawdown':
            target_cols = ['dca_max_drawdown', 'aous_max_drawdown', 'lump_sum_max_drawdown']
        elif optimization_target == 'sharpe':
            target_cols = ['dca_sharpe', 'aous_sharpe', 'lump_sum_sharpe']
        else:
            raise ValueError(f"Invalid optimization target: {optimization_target}")
        
        # Check if all required columns exist
        missing_cols = [col for col in target_cols if col not in results_df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return {}
            
        if risk_adjusted:
            # Calculate risk-adjusted score
            for col in target_cols:
                strategy = col.split('_')[0]
                drawdown_col = f'{strategy}_max_drawdown'
                if drawdown_col in results_df.columns:
                    results_df[f'{strategy}_score'] = (
                        results_df[col] / results_df[drawdown_col].abs()
                    )
            score_cols = [f'{col.split("_")[0]}_score' for col in target_cols]
        else:
            score_cols = target_cols

        optimal_params = {}
        for col in score_cols:
            if col not in results_df.columns:
                continue
                
            strategy = col.split('_')[0]
            try:
                best_idx = results_df[col].idxmax()
                optimal_params[strategy] = {
                    'base_investment': results_df.loc[best_idx, 'base_investment'],
                    'dip_investment': results_df.loc[best_idx, 'dip_investment'],
                    'dip_threshold': results_df.loc[best_idx, 'dip_threshold'],
                    'holding_period': results_df.loc[best_idx, 'holding_period'],
                    'metrics': {
                        'roi': results_df.loc[best_idx, f'{strategy}_roi'],
                        'max_drawdown': results_df.loc[best_idx, f'{strategy}_max_drawdown'],
                        'sharpe': results_df.loc[best_idx, f'{strategy}_sharpe'],
                        'btc_held': results_df.loc[best_idx, f'{strategy}_btc_held']
                    }
                }
            except Exception as e:
                print(f"Error finding optimal parameters for {strategy}: {str(e)}")
                continue

        return optimal_params

    def _calculate_sharpe_ratio(self, portfolio: pd.DataFrame, 
                              risk_free_rate: float = 0.04) -> float:
        """Calculate Sharpe ratio for a portfolio"""
        daily_returns = portfolio['portfolio_value'].pct_change().dropna()
        excess_returns = daily_returns - (risk_free_rate / 252)
        
        if len(excess_returns) == 0:
            return 0
            
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

# Example usage:
if __name__ == "__main__":
    # This would be imported from your backtest results
    # Example:
    # from btc_backtest_core import BacktestCore
    # backtest = BacktestCore('bitcoin_price_data.csv')
    
    # params = SimulationParams(
    #     base_investments=[50, 100, 200],
    #     dip_investments=[500, 1000, 2000],
    #     dip_thresholds=[0.05, 0.10, 0.15],
    #     holding_periods=[1, 7, 30]
    # )
    
    # simulator = ParameterSimulator(backtest)
    # results = simulator.run_parallel_simulations(params)
    # analysis_df = simulator.analyze_simulation_results(results)
    # optimal_params = simulator.find_optimal_parameters(analysis_df)
    pass