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
        This report analyzes three Bitcoin investment strategies: Dollar Cost Averaging (DCA), 
        Enhanced DCA with Dip Buying (Aous), and Lump Sum investing. Key findings include:
        <br/><br/>
        • Lump Sum achieved highest absolute returns (1,250% ROI) but with highest risk
        <br/>
        • DCA provided best risk-adjusted returns (Sharpe ratio: 2.01)
        <br/>
        • Aous strategy showed superior downside protection (Sortino ratio: 7.15)
        <br/>
        • All strategies performed well in bull markets, but DCA/Aous were more resilient in bear markets
        <br/>
        • Optimal strategy involves small base investments with aggressive dip-buying
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
        
        analysis = """
        <b>Investment Efficiency:</b>
        <br/>
        • DCA: $180,200 invested → $760,958 (322% ROI)
        <br/>
        • Aous: $1,052,200 invested → $4,424,237 (320% ROI)
        <br/>
        • Lump Sum: $365,000 invested → $4,930,894 (1,250% ROI)
        <br/><br/>
        <b>Capital Efficiency:</b>
        <br/>
        • DCA shows highest capital efficiency with similar ROI to Aous using much less capital
        <br/>
        • Lump Sum demonstrates power of early market entry but with higher risk
        <br/>
        • Aous strategy requires significant capital but provides better risk management
        """
        elements.append(Paragraph(analysis, self.custom_style))
        return elements

    def _create_risk_analysis(self):
        """Create risk analysis section"""
        elements = []
        elements.append(Paragraph("Risk Analysis", self.heading_style))
        
        analysis = f"""
        <b>Volatility Profile:</b>
        <br/>
        • DCA: {self.comparison_df.loc['Volatility (%)']['DCA']:.1f}% volatility with Sharpe ratio {self.comparison_df.loc['Sharpe Ratio']['DCA']:.2f}
        <br/>
        • Aous: {self.comparison_df.loc['Volatility (%)']['Aous']:.1f}% volatility with Sharpe ratio {self.comparison_df.loc['Sharpe Ratio']['Aous']:.2f}
        <br/>
        • Lump Sum: {self.comparison_df.loc['Volatility (%)']['Lump Sum']:.1f}% volatility with Sharpe ratio {self.comparison_df.loc['Sharpe Ratio']['Lump Sum']:.2f}
        <br/><br/>
        <b>Drawdown Profile:</b>
        <br/>
        • DCA max drawdown: {self.comparison_df.loc['Max Drawdown (%)']['DCA']:.1f}%
        <br/>
        • Aous max drawdown: {self.comparison_df.loc['Max Drawdown (%)']['Aous']:.1f}%
        <br/>
        • Lump Sum max drawdown: {self.comparison_df.loc['Max Drawdown (%)']['Lump Sum']:.1f}%
        """
        elements.append(Paragraph(analysis, self.custom_style))
        return elements

    def _create_yearly_performance(self):
        """Create yearly performance section"""
        elements = []
        elements.append(Paragraph("Yearly Performance", self.heading_style))
        
        # Create yearly returns table
        data = [['Year', 'DCA', 'Aous', 'Lump Sum']]
        years = sorted(self.analyses['DCA'].yearly_returns.keys())
        
        for year in years:
            data.append([
                str(year),
                f"{self.analyses['DCA'].yearly_returns[year]:.1f}%",
                f"{self.analyses['Aous'].yearly_returns[year]:.1f}%",
                f"{self.analyses['Lump Sum'].yearly_returns[year]:.1f}%"
            ])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
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
        elements.append(Paragraph("Optimal Strategy Parameters", self.heading_style))
        
        for strategy, params in self.optimal_params.items():
            param_text = f"""
            <b>{strategy.upper()}:</b>
            <br/>
            • Base Investment: ${params['base_investment']}
            <br/>
            • Dip Investment: ${params['dip_investment']}
            <br/>
            • Dip Threshold: {params['dip_threshold']*100}%
            <br/>
            • Holding Period: {params['holding_period']} days
            <br/>
            • ROI: {params['metrics']['roi']:.2f}%
            <br/>
            • Max Drawdown: {params['metrics']['max_drawdown']:.2f}%
            <br/><br/>
            """
            elements.append(Paragraph(param_text, self.custom_style))
        return elements

    def _create_visualizations(self):
        """Create and add visualizations to the report"""
        elements = []
        elements.append(Paragraph("Strategy Visualizations", self.heading_style))
        
        # Portfolio Values Plot
        plt.figure(figsize=(10, 6))
        for strategy, portfolio in self.portfolios.items():
            plt.plot(portfolio.index, portfolio['portfolio_value'], label=strategy)
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
            plt.plot(portfolio.index, drawdown, label=strategy)
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
        
        return elements

if __name__ == "__main__":
    # Example usage:
    # report_gen = ReportGenerator(portfolios, analyses, comparison_df, optimal_params)
    # report_gen.create_report("bitcoin_strategy_analysis.pdf")
    pass 