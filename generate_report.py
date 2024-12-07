from btc_backtest_core import BacktestCore
from btc_analyzer import BacktestAnalyzer
from btc_visualizer import BacktestVisualizer
from btc_parameter_sim import ParameterSimulator, SimulationParams
from btc_report_generator import ReportGenerator
import pandas as pd

def main():
    # Initialize components
    backtest = BacktestCore(
        'bitcoin_price_data_2020-01-01_to_2024-12-06.csv',
        price_col='price',
        date_col='date'
    )
    analyzer = BacktestAnalyzer()
    visualizer = BacktestVisualizer()

    # Run all strategies
    portfolios = {
        'DCA': backtest.run_dca_strategy(daily_investment=100),
        'Aous': backtest.run_aous_strategy(
            daily_investment=100,
            dip_investment=1000,
            dip_threshold=0.1,
            holding_period=30
        ),
        'Lump Sum': backtest.run_lump_sum_strategy(total_investment=365000),  # Equivalent to $100 daily for 10 years
        'RSI': backtest.run_rsi_strategy(
            base_investment=100,
            rsi_thresholds={30: 2000, 20: 5000}
        ),
        'MA Momentum': backtest.run_ma_momentum_strategy(
            base_investment=100,
            ma_multipliers={'MA20': 2, 'MA50': 3}
        )
    }

    # Analyze each strategy
    analyses = {}
    for name, portfolio in portfolios.items():
        analyses[name] = analyzer.analyze_strategy(portfolio, backtest.df)

    # Compare strategies
    comparison_df = analyzer.compare_strategies(portfolios)

    # Visualize results
    visualizer.plot_portfolio_values(portfolios, save_path='portfolio_values.png')
    visualizer.plot_btc_holdings(portfolios, save_path='btc_holdings.png')
    visualizer.plot_drawdowns(portfolios, save_path='drawdowns.png')
    visualizer.plot_investment_amounts(portfolios, save_path='investment_amounts.png')  # Make sure this is called
    
    try:
        # Create interactive dashboard
        visualizer.create_interactive_dashboard(portfolios, backtest.df)
    except Exception as e:
        print(f"Error creating dashboard: {str(e)}")

    # Run parameter simulation for RSI strategy
    rsi_params = SimulationParams(
        base_investments=[50, 100, 200],
        rsi_thresholds=[
            {30: 1000, 20: 3000},
            {30: 2000, 20: 5000},
            {25: 3000, 15: 7000}
        ],
        rsi_periods=[14, 21, 28]
    )

    # Run parameter simulation for MA strategy
    ma_params = SimulationParams(
        base_investments=[50, 100, 200],
        ma_multipliers=[
            {'MA20': 1.5, 'MA50': 2},
            {'MA20': 2, 'MA50': 3},
            {'MA20': 3, 'MA50': 4}
        ]
    )

    # Run parameter simulation for Aous strategy
    aous_params = SimulationParams(
        base_investments=[50, 100, 200],
        dip_investments=[500, 1000, 2000],
        dip_thresholds=[0.05, 0.10, 0.15],
        holding_periods=[1, 7, 30]
    )

    simulator = ParameterSimulator(backtest)
    
    # Combine all parameter simulations
    all_results = {}
    
    print("\nRunning RSI strategy simulations...")
    all_results['RSI'] = simulator.run_parallel_simulations(rsi_params)
    
    print("\nRunning MA strategy simulations...")
    all_results['MA'] = simulator.run_parallel_simulations(ma_params)
    
    print("\nRunning Aous strategy simulations...")
    all_results['Aous'] = simulator.run_parallel_simulations(aous_params)
    
    optimal_params = {}
    for strategy, results in all_results.items():
        if results:  # Only analyze if we have valid results
            analysis_df = simulator.analyze_simulation_results(results)
            if not analysis_df.empty:
                optimal_params[strategy] = simulator.find_optimal_parameters(
                    analysis_df,
                    optimization_target='roi',
                    risk_adjusted=True
                )

    # Generate PDF report
    report_gen = ReportGenerator(portfolios, analyses, comparison_df, optimal_params)
    report_gen.create_report("bitcoin_strategy_analysis.pdf")

    print("\nReport generated successfully: bitcoin_strategy_analysis.pdf")

if __name__ == "__main__":
    main() 