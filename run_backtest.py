from btc_backtest_core import BacktestCore
from btc_analyzer import BacktestAnalyzer
from btc_visualizer import BacktestVisualizer
from btc_parameter_sim import ParameterSimulator, SimulationParams
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
        print(f"\n{analyzer.generate_report(name, analyses[name])}")

    # Compare strategies
    comparison_df = analyzer.compare_strategies(portfolios)
    print("\nStrategy Comparison:")
    print(comparison_df)

    # Visualize results
    visualizer.plot_portfolio_values(portfolios, save_path='portfolio_values.png')
    visualizer.plot_btc_holdings(portfolios, save_path='btc_holdings.png')
    visualizer.plot_drawdowns(portfolios, save_path='drawdowns.png')
    
    try:
        # Create interactive dashboard
        visualizer.create_interactive_dashboard(portfolios, backtest.df)
    except Exception as e:
        print(f"Error creating dashboard: {str(e)}")

    # Run parameter simulation
    params = SimulationParams(
        base_investments=[50, 100, 200],
        dip_investments=[500, 1000, 2000],
        dip_thresholds=[0.05, 0.10, 0.15],
        holding_periods=[1, 7, 30]
    )

    try:
        simulator = ParameterSimulator(backtest)
        results = simulator.run_parallel_simulations(params)
        analysis_df = simulator.analyze_simulation_results(results)
        
        if not analysis_df.empty:
            optimal_params = simulator.find_optimal_parameters(
                analysis_df, 
                optimization_target='roi',
                risk_adjusted=True
            )
            
            if optimal_params:
                print("\nOptimal Parameters for Each Strategy:")
                for strategy, params in optimal_params.items():
                    print(f"\n{strategy.upper()}:")
                    print(f"Base Investment: ${params['base_investment']}")
                    print(f"Dip Investment: ${params['dip_investment']}")
                    print(f"Dip Threshold: {params['dip_threshold']*100}%")
                    print(f"Holding Period: {params['holding_period']} days")
                    print("Metrics:")
                    for metric, value in params['metrics'].items():
                        print(f"- {metric}: {value:.2f}")
        else:
            print("\nNo valid simulation results to analyze")
    except Exception as e:
        print(f"Error in parameter simulation: {str(e)}")

    # Export results
    for name, analysis in analyses.items():
        try:
            analyzer.export_results(analysis, f'{name.lower()}_results.json')
        except Exception as e:
            print(f"Error exporting results for {name}: {str(e)}")

if __name__ == "__main__":
    main()
