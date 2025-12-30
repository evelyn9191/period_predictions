import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import os
import argparse

class PeriodPredictor:
    def __init__(self, cycle_data: List[Tuple[datetime, datetime, int, int]]):
        self.cycle_data = cycle_data
        self.cycle_lengths = [cycle[2] for cycle in cycle_data]
        self.period_durations = [cycle[3] for cycle in cycle_data]
        self.last_period_start = cycle_data[0][0]
        
        self.cycle_mu = np.mean(self.cycle_lengths)
        self.cycle_sigma = np.std(self.cycle_lengths)
        self.period_mu = np.mean(self.period_durations)
        self.period_sigma = np.std(self.period_durations) if len(self.period_durations) > 1 else 0.5
        
    def simulate_future_cycles(self, n_simulations: int = 10000, forecast_months: int = 12) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        start_date = self.last_period_start
        end_date = start_date + timedelta(days=forecast_months * 31 + 60)
        
        date_range = pd.date_range(start=start_date, end=end_date)
        simulations = np.zeros((n_simulations, len(date_range)), dtype=bool)
        
        for sim in range(n_simulations):
            current_period_start = start_date
            
            while current_period_start <= end_date:
                cycle_length = max(20, int(np.random.normal(self.cycle_mu, self.cycle_sigma)))
                period_duration = max(3, int(np.random.normal(self.period_mu, self.period_sigma)))
                
                for day_offset in range(period_duration):
                    period_day = current_period_start + timedelta(days=day_offset)
                    if start_date <= period_day <= end_date:
                        idx = (date_range == period_day).argmax()
                        if idx < len(simulations[sim]):
                            simulations[sim, idx] = True
                
                current_period_start += timedelta(days=cycle_length)
        
        return date_range, simulations
    
    def calculate_probabilities(self, simulations: np.ndarray) -> np.ndarray:
        return simulations.mean(axis=0)

    def _create_heatmap(self, df: pd.DataFrame) -> go.Figure:
        df_future = df[df['date'] >= self.last_period_start].copy()
        df_future['year_month'] = df_future['date'].dt.to_period('M')
        months = df_future['year_month'].unique()[:12]
        
        n_months = min(12, len(months))
        fig = make_subplots(
            rows=4,
            cols=3,
            subplot_titles=[f"{m.strftime('%B %Y')}" for m in months[:n_months]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        for idx, month in enumerate(months[:n_months]):
            row = (idx // 3) + 1
            col = (idx % 3) + 1
            
            month_start = month.to_timestamp()
            month_end = (month + 1).to_timestamp() - timedelta(days=1)
            
            month_df = df_future[df_future['year_month'] == month].copy()
            
            first_weekday = month_start.weekday()
            days_in_month = (month_end - month_start).days + 1
            
            grid = np.full((6, 7), np.nan)
            day_labels = np.full((6, 7), '', dtype=object)
            hover_text = np.full((6, 7), '', dtype=object)
            
            for day in range(1, days_in_month + 1):
                current_date = month_start + timedelta(days=day - 1)
                weekday = current_date.weekday()
                week = (day + first_weekday - 1) // 7
                
                if week < 6:
                    prob_row = month_df[month_df['date'] == current_date]
                    if not prob_row.empty:
                        probability = prob_row['probability'].values[0]
                        grid[week, weekday] = probability
                        day_labels[week, weekday] = str(day)
                        hover_text[week, weekday] = f"{current_date.strftime('%a %b %d')}<br>Probability: {probability:.1%}"
                    else:
                        grid[week, weekday] = 0
                        day_labels[week, weekday] = str(day)
                        hover_text[week, weekday] = f"{current_date.strftime('%a %b %d')}<br>No data"
            
            heatmap = go.Heatmap(
                z=grid,
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                y=['', '', '', '', '', ''],
                colorscale=[
                    [0, '#ffffff'],
                    [0.05, '#fff5eb'],
                    [0.15, '#fee6ce'],
                    [0.30, '#fdd0a2'],
                    [0.45, '#fdae6b'],
                    [0.60, '#fd8d3c'],
                    [0.75, '#f16913'],
                    [0.90, '#d94801'],
                    [1.0, '#8c2d04']
                ],
                zmin=0,
                zmax=1,
                showscale=(idx == n_months - 1),
                text=day_labels,
                texttemplate='%{text}',
                textfont=dict(size=10),
                hovertext=hover_text,
                hoverinfo='text',
                colorbar=dict(
                    title='Probability',
                    x=1.02,
                    tickformat='.0%',
                    len=0.5
                ) if idx == n_months - 1 else None
            )
            
            fig.add_trace(heatmap, row=row, col=col)
        
        fig.update_xaxes(showticklabels=True, side='top')
        fig.update_yaxes(showticklabels=False)
        
        fig.update_layout(
            title='Period Probability Calendar - Next 12 Months',
            height=1000,
            showlegend=False,
            margin=dict(l=50, r=120, t=100, b=50)
        )
        
        return fig

    def _create_cycle_distribution_plot(self) -> go.Figure:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cycle Length Distribution', 'Period Duration Distribution'),
            horizontal_spacing=0.15
        )
        
        fig.add_trace(go.Histogram(
            x=self.cycle_lengths,
            name='Cycle Lengths',
            marker_color='#3498db',
            opacity=0.75,
            nbinsx=15,
            hovertemplate='%{x} days<br>Count: %{y}<extra></extra>'
        ), row=1, col=1)
        
        x_cycle = np.linspace(min(self.cycle_lengths) - 3, max(self.cycle_lengths) + 3, 100)
        pdf_cycle = stats.norm.pdf(x_cycle, self.cycle_mu, self.cycle_sigma) * len(self.cycle_lengths) * 2
        fig.add_trace(go.Scatter(
            x=x_cycle,
            y=pdf_cycle,
            mode='lines',
            name='Fitted Distribution',
            line=dict(color='#e74c3c', width=3)
        ), row=1, col=1)
        
        fig.add_trace(go.Histogram(
            x=self.period_durations,
            name='Period Durations',
            marker_color='#e74c3c',
            opacity=0.75,
            nbinsx=10,
            hovertemplate='%{x} days<br>Count: %{y}<extra></extra>'
        ), row=1, col=2)
        
        x_period = np.linspace(min(self.period_durations) - 1, max(self.period_durations) + 1, 100)
        pdf_period = stats.norm.pdf(x_period, self.period_mu, self.period_sigma) * len(self.period_durations) * 1.5
        fig.add_trace(go.Scatter(
            x=x_period,
            y=pdf_period,
            mode='lines',
            name='Fitted Distribution',
            line=dict(color='#3498db', width=3)
        ), row=1, col=2)
        
        fig.update_xaxes(title_text='Days', row=1, col=1)
        fig.update_xaxes(title_text='Days', row=1, col=2)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=2)
        
        fig.update_layout(
            title='Historical Data Analysis',
            showlegend=False,
            height=400,
            bargap=0.1
        )
        
        return fig
    
    def generate_html_report(self, date_range: pd.DatetimeIndex, probabilities: np.ndarray, output_file: str = 'period_prediction_report.html'):
        """Generate an interactive HTML report with period predictions."""
        # Create DataFrame with results
        df = pd.DataFrame({
            'date': date_range,
            'probability': probabilities,
            'month': date_range.month_name(),
            'day': date_range.day,
            'year': date_range.year
        })
        
        # Create figures
        heatmap_fig = self._create_heatmap(df)
        dist_fig = self._create_cycle_distribution_plot()
        
        high_risk_days = df[df['probability'] > 0.5].copy()
        high_risk_days['date_str'] = high_risk_days['date'].dt.strftime('%b %d, %Y')
        
        stats_text = f"""
        <h2>Cycle Statistics</h2>
        <ul>
            <li><strong>Average cycle length:</strong> {self.cycle_mu:.1f} ± {self.cycle_sigma:.1f} days</li>
            <li><strong>Cycle range:</strong> {min(self.cycle_lengths)} - {max(self.cycle_lengths)} days</li>
            <li><strong>Average period duration:</strong> {self.period_mu:.1f} ± {self.period_sigma:.1f} days</li>
            <li><strong>Period range:</strong> {min(self.period_durations)} - {max(self.period_durations)} days</li>
            <li><strong>Data points:</strong> {len(self.cycle_data)} cycles</li>
            <li><strong>Last period start:</strong> {self.last_period_start.strftime('%b %d, %Y')}</li>
        </ul>
        
        <h2>Interpretation Guide</h2>
        <ul>
            <li><strong>Probability &lt; 10%:</strong> Very unlikely - safe for planning</li>
            <li><strong>Probability 10-30%:</strong> Low risk - generally safe</li>
            <li><strong>Probability 30-50%:</strong> Moderate risk - consider backup plans</li>
            <li><strong>Probability 50-70%:</strong> High risk - likely to have period</li>
            <li><strong>Probability &gt; 70%:</strong> Very high risk - avoid if possible</li>
        </ul>
        
        <p><em>Note: Predictions become less certain further into the future due to natural cycle variability.</em></p>
        """
        
        # Create HTML report
        try:
            # Convert figures to JSON strings first to catch any serialization errors
            dist_json = dist_fig.to_json()
            heatmap_json = heatmap_fig.to_json()
            
            # Create HTML content with proper escaping
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Period Prediction Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .plot-container {{ margin: 30px 0; }}
                    .stats {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 30px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Period Prediction Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="stats">
                        {stats_text}
                    </div>
                    
                    <div class="plot-container">
                        <h2>Historical Cycle Analysis</h2>
                        <div id="dist-plot"></div>
                    </div>
                    
                    <div class="plot-container">
                        <h2>Period Probability Calendar Heatmap</h2>
                        <div id="heatmap"></div>
                    </div>
                </div>
                
                <script>
                    // Initialize with empty data
                    const distData = {dist_json};
                    const heatmapData = {heatmap_json};
                    
                    // Render plots after page loads
                    document.addEventListener('DOMContentLoaded', function() {{
                        try {{
                            Plotly.newPlot('dist-plot', distData.data, distData.layout);
                            Plotly.newPlot('heatmap', heatmapData.data, heatmapData.layout);
                        }} catch (e) {{
                            console.error('Error rendering plots:', e);
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            print(f"Error generating HTML report: {str(e)}")
            print("Please report this issue with the error message above.")
            raise
        
        print(f"Report generated: {os.path.abspath(output_file)}")
        return output_file
    

def parse_cycle_data(csv_file: str) -> List[Tuple[datetime, datetime, int, int]]:
    try:
        df = pd.read_csv(csv_file, parse_dates=['start_date', 'end_date'])
        df = df.dropna()
        df = df.sort_values('start_date', ascending=False)
        return [
            (row['start_date'].to_pydatetime(), 
             row['end_date'].to_pydatetime(), 
             int(row['cycle_length']),
             int(row['period_duration']))
            for _, row in df.iterrows()
        ]
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Predict future period probabilities.')
    parser.add_argument('--csv', type=str, default='cycle_data.csv',
                      help='Path to CSV file with cycle data')
    parser.add_argument('--output', type=str, default='period_prediction_report.html',
                      help='Output HTML report file')
    parser.add_argument('--simulations', type=int, default=10000,
                      help='Number of Monte Carlo simulations')
    args = parser.parse_args()
    
    print(f"Loading cycle data from {args.csv}...")
    cycle_data = parse_cycle_data(args.csv)
    
    if not cycle_data:
        print(f"No valid cycle data found in {args.csv}. Please check the file format.")
        print("Expected CSV format: start_date,end_date,cycle_length")
        return
        
    print(f"Loaded {len(cycle_data)} cycles.")
    
    predictor = PeriodPredictor(cycle_data)
    
    print(f"Running {args.simulations:,} Monte Carlo simulations...")
    date_range, simulations = predictor.simulate_future_cycles(n_simulations=args.simulations, forecast_months=12)
    
    print("Calculating probabilities...")
    probabilities = predictor.calculate_probabilities(simulations)
    
    print("Generating report...")
    report_file = predictor.generate_html_report(date_range, probabilities, args.output)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Cycle length: {predictor.cycle_mu:.1f} ± {predictor.cycle_sigma:.1f} days")
    print(f"Period duration: {predictor.period_mu:.1f} ± {predictor.period_sigma:.1f} days")
    print(f"\nReport saved to: {os.path.abspath(report_file)}")
    print(f"\nOpen in browser: file://{os.path.abspath(report_file)}")
    print("="*60)

if __name__ == "__main__":
    main()
