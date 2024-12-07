from btc_backtest_core import BacktestCore
from btc_analyzer import BacktestAnalyzer
from btc_visualizer import BacktestVisualizer
from btc_parameter_sim import ParameterSimulator, SimulationParams
from btc_report_generator import ReportGenerator
import pandas as pd

def main():
    # Initialize components
    backtest = BacktestCore('bitcoin_price_data_2020-01-01_to_2024-12-06.csv')
    analyzer = BacktestAnalyzer()
    visualizer = BacktestVisualizer()

    # Run basic strategies
    portfolios = {
        'DCA': backtest.run_dca_strategy(daily_investment=100),
        'Aous': backtest.run_aous_strategy(daily_investment=100, dip_investment=1000),
        'Lump Sum': backtest.run_lump_sum_strategy(total_investment=365000)  # Equivalent to $100 daily for 10 years
    }

    # Analyze each strategy
    analyses = {}
    for name, portfolio in portfolios.items():
        analyses[name] = analyzer.analyze_strategy(portfolio, backtest.df)

    # Compare strategies
    comparison_df = analyzer.compare_strategies(portfolios)

    # Run parameter simulation
    params = SimulationParams(
        base_investments=[50, 100, 200],
        dip_investments=[500, 1000, 2000],
        dip_thresholds=[0.05, 0.10, 0.15],
        holding_periods=[1, 7, 30]
    )

    simulator = ParameterSimulator(backtest)
    results = simulator.run_parallel_simulations(params)
    analysis_df = simulator.analyze_simulation_results(results)
    
    optimal_params = {}
    if not analysis_df.empty:
        optimal_params = simulator.find_optimal_parameters(
            analysis_df, 
            optimization_target='roi',
            risk_adjusted=True
        )

    # Generate PDF report
    report_gen = ReportGenerator(portfolios, analyses, comparison_df, optimal_params)
    report_gen.create_report("bitcoin_strategy_analysis.pdf")

    print("Report generated successfully: bitcoin_strategy_analysis.pdf")

if __name__ == "__main__":
    main() 