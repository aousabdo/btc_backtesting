import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import numpy as np
from datetime import datetime

class ReportGenerator:
    def __init__(self, portfolios, analyses, comparison_df, optimal_params):
        self.portfolios = portfolios
        self.analyses = analyses
        self.comparison_df = comparison_df
        self.optimal_params = optimal_params
        self.styles = getSampleStyleSheet()
        self.custom_style = ParagraphStyle(
            'CustomStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            spaceAfter=20
        )
        self.title_style = ParagraphStyle(
            'TitleStyle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        self.heading_style = ParagraphStyle(
            'HeadingStyle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20
        )

    def create_report(self, output_path: str):
        """Generate comprehensive PDF report"""
        doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        story = []

        # Title
        story.append(Paragraph("Bitcoin Investment Strategy Analysis Report", self.title_style))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.custom_style))
        story.append(Spacer(1, 20))

        # Executive Summary
        story.extend(self._create_executive_summary())
        story.append(Spacer(1, 20))

        # Strategy Comparison
        story.extend(self._create_strategy_comparison())
        story.append(Spacer(1, 20))

        # Performance Analysis
        story.extend(self._create_performance_analysis())
        story.append(Spacer(1, 20))

        # Risk Analysis
        story.extend(self._create_risk_analysis())
        story.append(Spacer(1, 20))

        # Yearly Performance
        story.extend(self._create_yearly_performance())
        story.append(Spacer(1, 20))

        # Drawdown Analysis
        story.extend(self._create_drawdown_analysis())
        story.append(Spacer(1, 20))

        # Optimal Parameters
        story.extend(self._create_optimal_parameters())
        story.append(Spacer(1, 20))

        # Save visualizations and add to report
        story.extend(self._create_visualizations())

        # Build the PDF
        doc.build(story)

    def _create_executive_summary(self):
        """Create executive summary section"""
        elements = []
        elements.append(Paragraph("Executive Summary", self.heading_style))
        
        summary = """
        This report analyzes five Bitcoin investment strategies: Dollar Cost Averaging (DCA), 
        Enhanced DCA with Dip Buying (Aous), Lump Sum investing, RSI-based investing, and Moving Average Momentum. Key findings include:
        <br/><br/>
        • Lump Sum achieved highest absolute returns but with highest risk
        <br/>
        • DCA provided consistent risk-adjusted returns
        <br/>
        • RSI strategy showed strong performance in volatile markets
        <br/>
        • MA Momentum captured larger positions during sustained downtrends
        <br/>
        • All strategies demonstrated unique risk-return profiles
        """
        elements.append(Paragraph(summary, self.custom_style))
        return elements

    def _create_strategy_comparison(self):
        """Create strategy comparison section"""
        elements = []
        elements.append(Paragraph("Strategy Comparison", self.heading_style))
        
        # Convert comparison DataFrame to table
        data = [['Metric'] + list(self.comparison_df.columns)]
        for idx, row in self.comparison_df.iterrows():
            data.append([idx] + [f"{x:,.2f}" for x in row])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(table)
        return elements

    def _create_performance_analysis(self):
        """Create performance analysis section"""
        elements = []
        elements.append(Paragraph("Performance Analysis", self.heading_style))
        
        analysis = f"""
        <b>Investment Efficiency:</b>
        <br/>
        • DCA: ${self.portfolios['DCA']['total_invested'].iloc[-1]:,.0f} invested → ${self.portfolios['DCA']['portfolio_value'].iloc[-1]:,.0f}
        <br/>
        • Aous: ${self.portfolios['Aous']['total_invested'].iloc[-1]:,.0f} invested → ${self.portfolios['Aous']['portfolio_value'].iloc[-1]:,.0f}
        <br/>
        • Lump Sum: ${self.portfolios['Lump Sum']['total_invested'].iloc[-1]:,.0f} invested → ${self.portfolios['Lump Sum']['portfolio_value'].iloc[-1]:,.0f}
        <br/>
        • RSI: ${self.portfolios['RSI']['total_invested'].iloc[-1]:,.0f} invested → ${self.portfolios['RSI']['portfolio_value'].iloc[-1]:,.0f}
        <br/>
        • MA Momentum: ${self.portfolios['MA Momentum']['total_invested'].iloc[-1]:,.0f} invested → ${self.portfolios['MA Momentum']['portfolio_value'].iloc[-1]:,.0f}
        <br/><br/>
        <b>Strategy Characteristics:</b>
        <br/>
        • DCA shows highest capital efficiency with consistent returns
        <br/>
        • RSI strategy effectively captures oversold conditions
        <br/>
        • MA Momentum adapts to market trends with dynamic position sizing
        <br/>
        • Aous combines regular investing with opportunistic dip buying
        <br/>
        • Lump Sum demonstrates power of early market entry
        """
        elements.append(Paragraph(analysis, self.custom_style))
        return elements

    def _create_risk_analysis(self):
        """Create risk analysis section"""
        elements = []
        elements.append(Paragraph("Risk Analysis", self.styles['Heading2']))
        
        # Introduction
        intro_text = "Below is a detailed risk analysis for each strategy, comparing key metrics:"
        elements.append(Paragraph(intro_text, self.styles['BodyText']))
        elements.append(Spacer(1, 12))
        
        # DCA Strategy
        elements.append(Paragraph("Dollar Cost Averaging (DCA):", self.styles['Heading3']))
        dca_text = """
        • Volatility: Low - Consistent investment regardless of market conditions
        • Max Drawdown: Moderate - Benefits from price averaging during downturns 
        • Risk-Adjusted Return: Medium - Stable but may miss opportunities"""
        elements.append(Paragraph(dca_text, self.styles['BodyText']))
        elements.append(Spacer(1, 12))
        
        # RSI Strategy
        elements.append(Paragraph("RSI Strategy:", self.styles['Heading3']))
        rsi_text = """
        • Volatility: Medium-High - Varies investment based on momentum
        • Max Drawdown: Lower in bear markets - Increases buying during oversold conditions
        • Risk-Adjusted Return: High - Optimizes entry points using momentum"""
        elements.append(Paragraph(rsi_text, self.styles['BodyText']))
        elements.append(Spacer(1, 12))
        
        # MA Strategy
        elements.append(Paragraph("MA Momentum Strategy:", self.styles['Heading3']))
        ma_text = """
        • Volatility: Medium - Follows trend-based signals
        • Max Drawdown: Medium - Uses trend following to avoid major downtrends
        • Risk-Adjusted Return: Medium-High - Benefits from trend identification"""
        elements.append(Paragraph(ma_text, self.styles['BodyText']))
        elements.append(Spacer(1, 12))
        
        # Aous Strategy
        elements.append(Paragraph("Aous Strategy:", self.styles['Heading3']))
        aous_text = """
        • Volatility: Medium-High - Combines multiple technical signals
        • Max Drawdown: Medium - Uses dip-buying to average down during corrections
        • Risk-Adjusted Return: High - Balances trend-following with mean reversion"""
        elements.append(Paragraph(aous_text, self.styles['BodyText']))
        elements.append(Spacer(1, 12))
        
        # Lump Sum Strategy
        elements.append(Paragraph("Lump Sum:", self.styles['Heading3']))
        lump_text = """
        • Volatility: Highest - Most exposed to entry point risk
        • Max Drawdown: Highest - No averaging benefit during market downturns
        • Risk-Adjusted Return: Varies significantly based on entry timing"""
        elements.append(Paragraph(lump_text, self.styles['BodyText']))
        elements.append(Spacer(1, 12))
        
        # Key Risk Considerations
        elements.append(Paragraph("Key Risk Considerations:", self.styles['Heading3']))
        risk_text = """
        • Market Timing Risk: Highest in Lump Sum, lowest in DCA
        • Technical Signal Risk: Present in RSI, MA, and Aous strategies
        • Psychological Risk: Highest in strategies with variable investment amounts
        • Execution Risk: Increases with strategy complexity (DCA lowest, Aous/RSI highest)"""
        elements.append(Paragraph(risk_text, self.styles['BodyText']))
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_yearly_performance(self):
        """Create yearly performance section"""
        elements = []
        elements.append(Paragraph("Yearly Performance", self.styles['Heading2']))
        
        # Extract yearly returns from analyses
        table_data = [['Year', 'DCA', 'Aous', 'Lump Sum', 'RSI', 'MA Momentum']]
        
        # Get all unique years from all strategies
        all_years = set()
        for strategy_name, analysis in self.analyses.items():
            if hasattr(analysis, 'yearly_returns'):
                all_years.update(analysis.yearly_returns.keys())
        
        # Sort years
        years = sorted(all_years)
        
        # Fill in yearly performance data
        for year in years:
            row = [str(year)]
            for strategy in ['DCA', 'Aous', 'Lump Sum', 'RSI', 'MA Momentum']:
                try:
                    value = self.analyses[strategy].yearly_returns.get(year, 'N/A')
                    if value != 'N/A':
                        row.append(f"{value:.1f}%")
                    else:
                        row.append('N/A')
                except (KeyError, AttributeError):
                    row.append('N/A')
            table_data.append(row)
        
        # Create table style
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOX', (0, 0), (-1, -1), 2, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.black),
        ])
        
        # Create table
        table = Table(table_data)
        table.setStyle(style)
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_drawdown_analysis(self):
        """Create drawdown analysis section"""
        elements = []
        elements.append(Paragraph("Drawdown Analysis", self.heading_style))
        
        analysis = """
        <b>Key Drawdown Periods:</b>
        <br/>
        • 2021-2023 Bear Market: All strategies experienced their maximum drawdowns
        <br/>
        • March 2020 COVID Crash: Quick recovery across all strategies
        <br/>
        • 2021 Mid-Year Correction: DCA and Aous showed better resilience
        <br/><br/>
        <b>Recovery Patterns:</b>
        <br/>
        • DCA/Aous strategies showed faster recovery due to continued buying
        <br/>
        • Lump Sum experienced longer recovery periods but higher absolute returns
        """
        elements.append(Paragraph(analysis, self.custom_style))
        return elements

    def _create_optimal_parameters(self):
        """Create optimal parameters section"""
        elements = []
        elements.append(Paragraph("Optimal Parameters", self.styles['Heading2']))
        
        # Process optimal parameters for each strategy
        for strategy, params in self.optimal_params.items():
            elements.append(Paragraph(f"{strategy} Strategy:", self.styles['Heading3']))
            
            param_lines = []
            metrics = params.get('metrics', {})
            
            # Add base investment if present
            if 'base_investment' in params:
                param_lines.append(f"• Base Investment: ${params['base_investment']}")
            
            # Strategy-specific parameters
            if strategy == 'RSI':
                if 'rsi_period' in params:
                    param_lines.append(f"• RSI Period: {params['rsi_period']}")
                if 'rsi_thresholds' in params:
                    param_lines.append(f"• RSI Thresholds: {params['rsi_thresholds']}")
            
            elif strategy == 'MA':
                if 'ma_multipliers' in params:
                    param_lines.append(f"• MA Multipliers: {params['ma_multipliers']}")
                if 'short_window' in params:
                    param_lines.append(f"• Short Window: {params['short_window']}")
                if 'long_window' in params:
                    param_lines.append(f"• Long Window: {params['long_window']}")
            
            elif strategy == 'Aous':
                if 'dip_investment' in params:
                    param_lines.append(f"• Dip Investment: ${params['dip_investment']}")
                if 'dip_threshold' in params:
                    param_lines.append(f"• Dip Threshold: {params['dip_threshold']*100:.2f}%")
                if 'holding_period' in params:
                    param_lines.append(f"• Holding Period: {params['holding_period']} days")
            
            # Add metrics if present
            if metrics:
                param_lines.append("\nPerformance Metrics:")
                if 'roi' in metrics:
                    param_lines.append(f"• ROI: {metrics['roi']:.2f}%")
                if 'max_drawdown' in metrics:
                    param_lines.append(f"• Max Drawdown: {metrics['max_drawdown']:.2f}%")
                if 'sharpe_ratio' in metrics:
                    param_lines.append(f"• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            
            # Join all lines with proper spacing
            param_text = '\n'.join(param_lines)
            elements.append(Paragraph(param_text, self.styles['BodyText']))
            elements.append(Spacer(1, 20))  # Add more space between strategies
        
        return elements

    def _create_visualizations(self):
        """Create and add visualizations to the report"""
        elements = []
        elements.append(Paragraph("Strategy Visualizations", self.heading_style))
        
        # Portfolio Values Plot
        plt.figure(figsize=(10, 6))
        colors = {
            'DCA': '#1f77b4',
            'Aous': '#ff7f0e',
            'Lump Sum': '#2ca02c',
            'RSI': '#d62728',
            'MA Momentum': '#9467bd'
        }
        
        for strategy, portfolio in self.portfolios.items():
            plt.plot(portfolio.index, portfolio['portfolio_value'], 
                    label=strategy, color=colors.get(strategy))
        
        plt.title('Portfolio Values Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        elements.append(Image(img_data, width=6*inch, height=4*inch))
        plt.close()
        
        # Drawdowns Plot
        plt.figure(figsize=(10, 6))
        for strategy, portfolio in self.portfolios.items():
            rolling_max = portfolio['portfolio_value'].cummax()
            drawdown = (portfolio['portfolio_value'] - rolling_max) / rolling_max * 100
            plt.plot(portfolio.index, drawdown, label=strategy, color=colors.get(strategy))
        
        plt.title('Portfolio Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        elements.append(Image(img_data, width=6*inch, height=4*inch))
        plt.close()
        
        # Investment Amounts Over Time
        plt.figure(figsize=(10, 6))
        for strategy, portfolio in self.portfolios.items():
            plt.plot(portfolio.index, portfolio['investment'].rolling(window=30).mean(), 
                    label=f"{strategy} (30-day avg)", color=colors.get(strategy))
        
        plt.title('Investment Amounts Over Time (30-day Moving Average)')
        plt.xlabel('Date')
        plt.ylabel('Investment Amount ($)')
        plt.yscale('log')
        # plt.ylim(bottom=.1)
        plt.legend()
        plt.grid(True)
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        elements.append(Image(img_data, width=6*inch, height=4*inch))
        plt.close()
        
        return elements

    def _create_strategy_descriptions(self):
        elements = []
        elements.append(Paragraph("Strategy Descriptions", self.styles['Heading2']))
        
        # DCA Strategy
        elements.append(Paragraph("Dollar Cost Averaging (DCA)", self.styles['Heading3']))
        dca_desc = """
        The Dollar Cost Averaging strategy involves investing a fixed amount of money at regular intervals, 
        regardless of the market price. This strategy helps reduce the impact of volatility and eliminates 
        the need to time the market. It's a passive, long-term investment approach that can help build 
        wealth steadily over time.
        """
        elements.append(Paragraph(dca_desc, self.styles['BodyText']))
        elements.append(Spacer(1, 12))
        
        # RSI Strategy
        elements.append(Paragraph("Relative Strength Index (RSI)", self.styles['Heading3']))
        rsi_desc = """
        The RSI strategy uses the Relative Strength Index, a momentum indicator that measures the speed 
        and magnitude of recent price changes. The strategy increases investment amounts when the RSI 
        indicates oversold conditions (low RSI values) and reduces investments during overbought 
        conditions (high RSI values). This approach aims to buy more aggressively during market dips 
        and less during potential market tops.
        
        Key Components:
        • RSI Period: Number of periods used to calculate the RSI
        • RSI Thresholds: Different investment levels based on RSI values
        • Base Investment: Minimum regular investment amount
        """
        elements.append(Paragraph(rsi_desc, self.styles['BodyText']))
        elements.append(Spacer(1, 12))
        
        # MA Strategy
        elements.append(Paragraph("Moving Average (MA) Momentum", self.styles['Heading3']))
        ma_desc = """
        The Moving Average Momentum strategy uses multiple moving averages to identify trends and 
        potential entry points. It adjusts investment amounts based on the relative positions of 
        different moving averages, investing more aggressively when shorter-term averages cross 
        above longer-term averages (bullish signals) and less during bearish conditions.
        
        Key Components:
        • Short Window: Period for short-term moving average
        • Long Window: Period for long-term moving average
        • MA Multipliers: Investment multipliers based on MA crossovers
        """
        elements.append(Paragraph(ma_desc, self.styles['BodyText']))
        elements.append(Spacer(1, 12))
        
        # Aous Strategy
        elements.append(Paragraph("Aous Strategy", self.styles['Heading3']))
        aous_desc = """
        The Aous strategy combines multiple technical indicators and market conditions to make 
        investment decisions. It incorporates both trend-following and mean-reversion elements, 
        using dip-buying during market corrections while maintaining a base investment level.
        
        Key Components:
        • Base Investment: Regular investment amount
        • Dip Investment: Additional investment during market dips
        • Dip Threshold: Percentage drop to trigger additional investments
        • Holding Period: Minimum time to hold positions
        """
        elements.append(Paragraph(aous_desc, self.styles['BodyText']))
        elements.append(Spacer(1, 20))
        
        return elements

if __name__ == "__main__":
    # Example usage:
    # report_gen = ReportGenerator(portfolios, analyses, comparison_df, optimal_params)
    # report_gen.create_report("bitcoin_strategy_analysis.pdf")
    pass 