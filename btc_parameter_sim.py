from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from btc_analyzer import BacktestAnalyzer

class SimulationParams:
    def __init__(self, 
                 base_investments: List[float] = None,
                 dip_investments: List[float] = None,
                 dip_thresholds: List[float] = None,
                 holding_periods: List[int] = None,
                 rsi_thresholds: List[dict] = None,
                 rsi_periods: List[int] = None,
                 ma_multipliers: List[dict] = None):
        
        self.base_investments = base_investments or [100]
        self.dip_investments = dip_investments or [1000]
        self.dip_thresholds = dip_thresholds or [0.1]
        self.holding_periods = holding_periods or [30]
        self.rsi_thresholds = rsi_thresholds or [{30: 2000, 20: 5000}]
        self.rsi_periods = rsi_periods or [14]
        self.ma_multipliers = ma_multipliers or [{'MA20': 2, 'MA50': 3}]

class ParameterSimulator:
    def __init__(self, backtest_core):
        self.backtest = backtest_core
        
    def generate_parameter_combinations(self, params: SimulationParams) -> List[dict]:
        """Generate all possible parameter combinations"""
        combinations = []
        
        # Original Aous strategy parameters
        if all([params.base_investments, params.dip_investments, params.dip_thresholds]):
            for base in params.base_investments:
                for dip in params.dip_investments:
                    for threshold in params.dip_thresholds:
                        for period in params.holding_periods:
                            combinations.append({
                                'strategy': 'aous',
                                'base_investment': base,
                                'dip_investment': dip,
                                'dip_threshold': threshold,
                                'holding_period': period
                            })
        
        # RSI strategy parameters
        if params.rsi_thresholds:
            for base in params.base_investments:
                for thresholds in params.rsi_thresholds:
                    for period in params.rsi_periods:
                        combinations.append({
                            'strategy': 'rsi',
                            'base_investment': base,
                            'rsi_thresholds': thresholds,
                            'rsi_period': period
                        })
        
        # MA strategy parameters
        if params.ma_multipliers:
            for base in params.base_investments:
                for multipliers in params.ma_multipliers:
                    combinations.append({
                        'strategy': 'ma',
                        'base_investment': base,
                        'ma_multipliers': multipliers
                    })
        
        return combinations

    def run_simulation(self, params: dict) -> Tuple[dict, pd.DataFrame]:
        """Run a single simulation with given parameters"""
        try:
            if params['strategy'] == 'aous':
                portfolio = self.backtest.run_aous_strategy(
                    daily_investment=params['base_investment'],
                    dip_investment=params['dip_investment'],
                    dip_threshold=params['dip_threshold'],
                    holding_period=params['holding_period']
                )
            elif params['strategy'] == 'rsi':
                portfolio = self.backtest.run_rsi_strategy(
                    base_investment=params['base_investment'],
                    rsi_thresholds=params['rsi_thresholds'],
                    rsi_period=params['rsi_period']
                )
            elif params['strategy'] == 'ma':
                portfolio = self.backtest.run_ma_momentum_strategy(
                    base_investment=params['base_investment'],
                    ma_multipliers=params['ma_multipliers']
                )
            else:
                raise ValueError(f"Unknown strategy: {params['strategy']}")
            
            return params, portfolio
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            return None

    def run_parallel_simulations(self, params: SimulationParams) -> List[Tuple[dict, pd.DataFrame]]:
        """Run simulations in parallel"""
        combinations = self.generate_parameter_combinations(params)
        
        with Pool() as pool:
            results = list(tqdm(
                pool.imap(self.run_simulation, combinations),
                total=len(combinations),
                desc="Running simulations"
            ))
        
        # Filter out None results from failed simulations
        return [r for r in results if r is not None]

    def analyze_simulation_results(self, results: List[Tuple[dict, pd.DataFrame]]) -> pd.DataFrame:
        """Analyze results from parallel simulations"""
        analyzer = BacktestAnalyzer()
        analysis_rows = []
        
        for params, portfolio in results:
            analysis = analyzer.analyze_strategy(portfolio, self.backtest.df)
            
            row = {
                'strategy': params['strategy'],
                'base_investment': params['base_investment'],
                'total_invested': portfolio['total_invested'].iloc[-1],
                'portfolio_value': portfolio['portfolio_value'].iloc[-1],
                'roi': analysis.roi,
                'volatility': analysis.volatility,
                'sharpe_ratio': analysis.sharpe_ratio,
                'sortino_ratio': analysis.sortino_ratio,
                'max_drawdown': analysis.max_drawdown
            }
            
            # Add strategy-specific parameters
            if params['strategy'] == 'aous':
                row.update({
                    'dip_investment': params['dip_investment'],
                    'dip_threshold': params['dip_threshold'],
                    'holding_period': params['holding_period']
                })
            elif params['strategy'] == 'rsi':
                row.update({
                    'rsi_thresholds': str(params['rsi_thresholds']),
                    'rsi_period': params['rsi_period']
                })
            elif params['strategy'] == 'ma':
                row.update({
                    'ma_multipliers': str(params['ma_multipliers']),
                })
            
            analysis_rows.append(row)
        
        return pd.DataFrame(analysis_rows)

    def find_optimal_parameters(self, analysis_df: pd.DataFrame,
                              optimization_target: str = 'roi',
                              risk_adjusted: bool = True) -> dict:
        """Find optimal parameters based on target metric"""
        if risk_adjusted and optimization_target == 'roi':
            analysis_df['score'] = analysis_df['roi'] * analysis_df['sharpe_ratio']
        else:
            analysis_df['score'] = analysis_df[optimization_target]
        
        best_idx = analysis_df['score'].idxmax()
        best_params = analysis_df.loc[best_idx].to_dict()
        
        # Clean up the parameters based on strategy
        strategy = best_params['strategy']
        metrics = {
            'roi': best_params['roi'],
            'sharpe_ratio': best_params['sharpe_ratio'],
            'sortino_ratio': best_params['sortino_ratio'],
            'max_drawdown': best_params['max_drawdown'],
            'volatility': best_params['volatility']
        }
        
        if strategy == 'aous':
            return {
                'base_investment': best_params['base_investment'],
                'dip_investment': best_params['dip_investment'],
                'dip_threshold': best_params['dip_threshold'],
                'holding_period': best_params['holding_period'],
                'metrics': metrics
            }
        elif strategy == 'rsi':
            return {
                'base_investment': best_params['base_investment'],
                'rsi_thresholds': eval(best_params['rsi_thresholds']),
                'rsi_period': best_params['rsi_period'],
                'metrics': metrics
            }
        elif strategy == 'ma':
            return {
                'base_investment': best_params['base_investment'],
                'ma_multipliers': eval(best_params['ma_multipliers']),
                'metrics': metrics
            }
        
        return best_params