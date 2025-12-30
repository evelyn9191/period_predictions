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
    def __init__(self, cycle_data: List[Tuple[datetime, datetime, int]]):
        """
        Initialize the PeriodPredictor with historical cycle data.
        
        Args:
            cycle_data: List of tuples containing (start_date, end_date, cycle_length)
        """
        self.cycle_data = cycle_data
        self.cycle_lengths = [cycle[2] for cycle in cycle_data]
        self.last_period_start = cycle_data[-1][0]  # Most recent period start
        
        # Fit a normal distribution to cycle lengths
        self.mu, self.sigma = np.mean(self.cycle_lengths), np.std(self.cycle_lengths)
        
    def simulate_future_cycles(self, n_simulations: int = 10000, forecast_years: int = 2) -> np.ndarray:
        """
        Simulate future cycles using the historical cycle length distribution.
        
        Args:
            n_simulations: Number of simulations to run
            forecast_years: Number of years to forecast into the future
            
        Returns:
            A 2D array where each row represents one simulation and each column is a day
        """
        # Create date range for the forecast period until the end of 2026
        end_date = datetime(2026, 12, 31)
        date_range = pd.date_range(
            start=self.last_period_start,
            end=end_date
        )
        forecast_days = len(date_range)  # Update forecast_days to match the actual range
        
        # Initialize array to store simulation results
        simulations = np.zeros((n_simulations, len(date_range)), dtype=bool)
        
        for sim in range(n_simulations):
            current_date = self.last_period_start
            period_day = 0
            cycle_length = int(np.random.normal(self.mu, self.sigma))
            
            while current_date < date_range[-1]:
                # For each day in the period (assuming 5 days per period)
                for day in range(5):
                    if current_date in date_range:
                        idx = (date_range == current_date).argmax()
                        simulations[sim, idx] = True
                    current_date += timedelta(days=1)
                
                # Move to next period
                cycle_length = int(np.random.normal(self.mu, self.sigma))
                current_date += timedelta(days=max(1, cycle_length - 5))  # Ensure we don't get stuck
        
        return date_range, simulations
    
    def calculate_probabilities(self, simulations: np.ndarray) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """
        Calculate daily probabilities from simulation results.
        
        Args:
            simulations: 2D array of simulation results
            
        Returns:
            Tuple of (dates, probabilities)
        """
        return simulations.mean(axis=0)

    def _create_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create an interactive heatmap of period probabilities for 2026."""
        # Filter for 2026
        df_2026 = df[df['date'].dt.year == 2026].copy()
        
        # Create a figure with subplots for each month
        months = df_2026['date'].dt.month.unique()
        n_months = len(months)
        
        # Calculate dynamic vertical spacing based on number of months
        vertical_spacing = max(0.02, min(0.1, 0.7 / n_months))
        
        fig = make_subplots(
            rows=n_months, 
            cols=1,
            subplot_titles=[f"{datetime(2026, month, 1).strftime('%B %Y')}" for month in months],
            vertical_spacing=vertical_spacing,
            row_heights=[1] * n_months
        )
        
        for i, month in enumerate(months, 1):
            # Filter data for the month
            month_df = df_2026[df_2026['date'].dt.month == month].copy()
            
            # Create a grid of dates for the month
            dates = pd.date_range(start=month_df['date'].min(), end=month_df['date'].max())
            days = [d.day for d in dates]
            
            # Create heatmap for the month
            heatmap = go.Heatmap(
                x=dates,
                y=['Probability'],
                z=[month_df['probability']],
                colorscale='YlOrRd',
                zmin=0,
                zmax=1,
                hoverinfo='x+z',
                hovertemplate='%{x|%b %d, %Y}<br>Probability: %{z:.1%}<extra></extra>',
                showscale=False
            )
            
            fig.add_trace(heatmap, row=i, col=1)
            
            # Add day numbers as annotations
            for j, day in enumerate(days):
                fig.add_annotation(
                    x=dates[j],
                    y=0.5,
                    text=str(day),
                    showarrow=False,
                    font=dict(
                        size=10,
                        color='black'
                    ),
                    row=i,
                    col=1
                )
        
        # Update layout
        fig.update_layout(
            title='2026 Daily Period Probability Heatmap',
            height=150 * n_months,
            showlegend=False,
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Update x-axes to show day numbers
        for i in range(1, n_months + 1):
            fig.update_xaxes(
                tickmode='array',
                tickvals=dates,
                ticktext=days,
                tickangle=0,
                row=i,
                col=1
            )
            
            # Add month separator lines
            if i < n_months:
                fig.add_hline(
                    y=1 - (i * (1/n_months)),
                    line=dict(color='black', width=1),
                    row=i,
                    col=1
                )

        return fig

    def _create_cycle_distribution_plot(self) -> go.Figure:
        """Create a histogram of cycle lengths."""
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=self.cycle_lengths,
            name='Cycle Lengths',
            marker_color='#EF553B',
            opacity=0.75,
            hovertemplate='%{x} days<br>Count: %{y}<extra></extra>'
        ))
        
        # Add normal distribution curve
        x = np.linspace(min(self.cycle_lengths) - 2, max(self.cycle_lengths) + 2, 100)
        pdf = stats.norm.pdf(x, self.mu, self.sigma) * len(self.cycle_lengths) * (max(self.cycle_lengths) - min(self.cycle_lengths)) / 20
        
        fig.add_trace(go.Scatter(
            x=x,
            y=pdf,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='#2E86C1', width=2)
        ))
        
        fig.update_layout(
            title='Cycle Length Distribution',
            xaxis_title='Cycle Length (days)',
            yaxis_title='Count',
            showlegend=True,
            bargap=0.1,
            height=500
        )
        
        return fig
    
    def _create_calendar_view(self, df: pd.DataFrame) -> go.Figure:
        """Create a calendar view of period probabilities for 2026."""
        # Filter for 2026
        df_2026 = df[df['date'].dt.year == 2026].copy()
        
        # Create a figure with subplots for each month
        months = df_2026['date'].dt.month.unique()
        n_months = len(months)
        
        # Calculate dynamic vertical spacing based on number of months
        vertical_spacing = max(0.02, min(0.1, 0.7 / n_months))
        
        fig = make_subplots(
            rows=n_months, 
            cols=1,
            subplot_titles=[f"{datetime(2026, month, 1).strftime('%B %Y')}" for month in months],
            vertical_spacing=vertical_spacing,
            row_heights=[1] * n_months
        )
        
        for i, month in enumerate(months, 1):
            # Filter data for the month
            month_df = df_2026[df_2026['date'].dt.month == month].copy()
            
            # Create a pivot table for the calendar view
            month_df['day'] = month_df['date'].dt.day
            month_df['week'] = month_df['date'].dt.isocalendar().week
            month_df['weekday'] = month_df['date'].dt.weekday  # Monday=0, Sunday=6
            
            # Get first and last day of the month
            first_day = month_df['date'].min()
            last_day = month_df['date'].max()
            
            # Create a grid of dates for the month
            dates = pd.date_range(start=first_day, end=last_day)
            grid = pd.DataFrame(index=range(6), columns=range(7))
            
            # Fill the grid with probabilities
            for date in dates:
                weekday = date.weekday()
                week_num = (date.day + first_day.day - 1) // 7
                if week_num < 6:  # Safety check
                    prob = month_df[month_df['date'] == date]['probability'].values
                    if len(prob) > 0:
                        grid.iloc[week_num, weekday] = prob[0]
            
            # Create heatmap for the month
            heatmap = go.Heatmap(
                z=grid.values,
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                y=['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
                colorscale='YlOrRd',
                zmin=0,
                zmax=1,
                showscale=False,
                hovertemplate='%{x} %{y}<br>Probability: %{z:.1%}<extra></extra>',
                texttemplate='%{z:.0%}',
                textfont={"size": 10}
            )
            
            fig.add_trace(heatmap, row=i, col=1)
        
        # Update layout
        fig.update_layout(
            title='2026 Monthly Calendar View',
            height=250 * n_months,
            showlegend=False,
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Update y-axis to show week numbers
        for i in range(1, n_months + 1):
            fig.update_yaxes(title_text=f"", row=i, col=1, tickmode='array', 
                           tickvals=[0, 1, 2, 3, 4, 5], 
                           ticktext=['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'])
        
        return fig
        
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
        calendar_fig = self._create_calendar_view(df)
        
        # Calculate statistics
        stats_text = f"""
        <h2>Cycle Statistics</h2>
        <ul>
            <li>Average cycle length: {self.mu:.1f} days</li>
            <li>Standard deviation: {self.sigma:.1f} days</li>
            <li>Shortest cycle: {min(self.cycle_lengths)} days</li>
            <li>Longest cycle: {max(self.cycle_lengths)} days</li>
            <li>Next predicted period start: {self._predict_next_period()}</li>
        </ul>
        """
        
        # Create HTML report
        try:
            # Convert figures to JSON strings first to catch any serialization errors
            dist_json = dist_fig.to_json()
            heatmap_json = heatmap_fig.to_json()
            calendar_json = calendar_fig.to_json()
            
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
                        <h2>Cycle Length Distribution</h2>
                        <div id="dist-plot"></div>
                    </div>
                    
                    <div class="plot-container">
                        <h2>Period Probability Heatmap</h2>
                        <div id="heatmap"></div>
                    </div>
                    
                    <div class="plot-container">
                        <h2>Monthly Calendar View</h2>
                        <div id="calendar"></div>
                    </div>
                </div>
                
                <script>
                    // Initialize with empty data
                    const distData = {dist_json};
                    const heatmapData = {heatmap_json};
                    const calendarData = {calendar_json};
                    
                    // Render plots after page loads
                    document.addEventListener('DOMContentLoaded', function() {{
                        try {{
                            Plotly.newPlot('dist-plot', distData.data, distData.layout);
                            Plotly.newPlot('heatmap', heatmapData.data, heatmapData.layout);
                            Plotly.newPlot('calendar', calendarData.data, calendarData.layout);
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
    
    def _predict_next_period(self) -> str:
        """Predict the next period start date with confidence interval."""
        # Calculate the next period start based on the last period start
        next_start = self.last_period_start + timedelta(days=int(self.mu))
        
        # If the next start is before 2026, project it into 2026
        if next_start.year < 2026:
            # Calculate how many cycles we need to add to get to 2026
            days_to_2026 = (datetime(2026, 1, 1) - self.last_period_start).days
            cycles_to_2026 = max(1, int(days_to_2026 / self.mu))
            next_start = self.last_period_start + timedelta(days=int(cycles_to_2026 * self.mu))
        
        # Ensure we're in 2026
        if next_start.year < 2026:
            next_start = datetime(2026, 1, 1) + timedelta(days=int(self.mu))
            
        lower_bound = next_start - timedelta(days=int(self.sigma))
        upper_bound = next_start + timedelta(days=int(self.sigma))
        
        # Format the date range
        if next_start.year == 2026 and lower_bound.year == 2026 and upper_bound.year == 2026:
            return f"{next_start.strftime('%b %d')} (likely between {lower_bound.strftime('%b %d')} and {upper_bound.strftime('%b %d')})"
        else:
            return f"{next_start.strftime('%b %d, %Y')} (likely between {lower_bound.strftime('%b %d, %Y')} and {upper_bound.strftime('%b %d, %Y')})"

def parse_cycle_data(csv_file: str) -> List[Tuple[datetime, datetime, int]]:
    """Parse cycle data from a CSV file.
    
    Args:
        csv_file: Path to the CSV file containing cycle data
        
    Returns:
        List of tuples (start_date, end_date, cycle_length)
    """
    try:
        df = pd.read_csv(csv_file, parse_dates=['start_date', 'end_date'])
        return [
            (row['start_date'].to_pydatetime(), 
             row['end_date'].to_pydatetime(), 
             int(row['cycle_length']))
            for _, row in df.iterrows()
        ]
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict future period probabilities.')
    parser.add_argument('--csv', type=str, default='cycle_data.csv',
                      help='Path to CSV file with cycle data')
    parser.add_argument('--output', type=str, default='period_prediction_report.html',
                      help='Output HTML report file')
    args = parser.parse_args()
    
    # Load cycle data from CSV
    print(f"Loading cycle data from {args.csv}...")
    cycle_data = parse_cycle_data(args.csv)
    
    if not cycle_data:
        print(f"No valid cycle data found in {args.csv}. Please check the file format.")
        print("Expected CSV format: start_date,end_date,cycle_length")
        return
        
    print(f"Loaded {len(cycle_data)} cycles.")
    
    # Initialize predictor
    predictor = PeriodPredictor(cycle_data)
    
    # Run simulations until the end of 2026
    print("Running simulations until the end of 2026...")
    date_range, simulations = predictor.simulate_future_cycles(n_simulations=10000, forecast_years=2)
    
    # Calculate probabilities
    print("Calculating probabilities...")
    probabilities = predictor.calculate_probabilities(simulations)
    
    # Generate HTML report
    print("Generating report...")
    report_file = predictor.generate_html_report(date_range, probabilities, args.output)
    
    print("\nAnalysis complete!")
    print(f"- Average cycle length: {predictor.mu:.1f} days")
    print(f"- Next predicted period: {predictor._predict_next_period()}")
    print(f"\nOpen the following file in your web browser to view the report:")
    print(f"file://{os.path.abspath(report_file)}")

if __name__ == "__main__":
    main()
