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

    def split_by_month(self, date_range):
        months = {}
        for i, d in enumerate(date_range):
            key = d.strftime("%Y-%m")
            months.setdefault(key, []).append(i)
        return months

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

    def create_monthly_scatter(
            self,
            date_range,
            simulations,
            probabilities,
            month_indices,
            title
    ):
        xs, ys = [], []

        for sim_idx in range(simulations.shape[0]):
            sim_days = np.where(simulations[sim_idx, month_indices])[0]
            xs.extend(date_range[month_indices][sim_days])
            ys.extend(
                sim_idx + np.random.uniform(-0.3, 0.3, size=len(sim_days))
            )

        fig = go.Figure()

        fig.add_trace(go.Scattergl(
            x=xs,
            y=ys,
            mode='markers',
            marker=dict(size=2, color='rgba(255, 80, 80, 0.25)'),
            name='Simulated periods'
        ))

        fig.add_trace(go.Scatter(
            x=date_range[month_indices],
            y=probabilities[month_indices] * simulations.shape[0],
            mode='lines',
            line=dict(color='black', width=2),
            name='Daily probability'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Simulation index / probability scale",
            height=350,
            showlegend=False
        )

        return fig

    def _create_scatter_plot(
            self,
            date_range: pd.DatetimeIndex,
            simulations: np.ndarray,
            probabilities: np.ndarray
    ) -> go.Figure:

        # Extract dot cloud
        xs = []
        ys = []

        for sim_idx in range(simulations.shape[0]):
            sim_days = np.where(simulations[sim_idx])[0]
            xs.extend(date_range[sim_days])
            ys.extend([sim_idx] * len(sim_days))

        scatter = go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=3,
                opacity=0.05,
                color="#e74c3c"
            ),
            name="Monte Carlo simulations",
            hoverinfo="skip"
        )

        # Probability curve (scaled for visibility)
        prob_line = go.Scatter(
            x=date_range,
            y=probabilities * simulations.shape[0],
            mode="lines",
            line=dict(color="#2c3e50", width=3),
            name="Daily probability (scaled)"
        )

        fig = go.Figure(data=[scatter, prob_line])

        fig.update_layout(
            title="Monte Carlo Period Prediction (Dot Density = Likelihood)",
            xaxis_title="Date",
            yaxis_title="Simulation index / probability scale",
            height=600,
            showlegend=True
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

    def generate_html_report(
            self,
            date_range: pd.DatetimeIndex,
            simulations: np.ndarray,
            probabilities: np.ndarray,
            output_file: str = "period_prediction_report.html",
    ):
        import numpy as np
        import plotly.graph_objects as go
        from plotly.io import to_html
        from datetime import datetime

        # --------------------------------------------------
        # 1. Split date indices by calendar month
        # --------------------------------------------------
        months = {}
        for i, d in enumerate(date_range):
            key = d.strftime("%Y-%m")
            months.setdefault(key, []).append(i)

        figures = []
        current_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # --------------------------------------------------
        # 2. Create one figure per month
        # --------------------------------------------------
        for month, indices in months.items():
            indices = np.array(indices)
            month_date = datetime.strptime(month + '-01', '%Y-%m-%d')
            
            # Skip if this is a past month (but not the current month)
            if month_date < current_month and month_date.month != current_month.month:
                continue
                
            xs, ys = [], []

            # Monte Carlo scatter
            for sim_idx in range(simulations.shape[0]):
                sim_days = np.where(simulations[sim_idx, indices])[0]
                xs.extend(date_range[indices][sim_days])
                ys.extend(sim_idx + np.random.uniform(-0.3, 0.3, size=len(sim_days)))

            fig = go.Figure()

            fig.add_trace(go.Scattergl(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    size=2,
                    color="rgba(255, 80, 80, 0.25)"
                ),
                name="Simulated periods"
            ))

            # Probability line (scaled to simulation count)
            fig.add_trace(go.Scatter(
                x=date_range[indices],
                y=probabilities[indices] * simulations.shape[0],
                mode="lines",
                line=dict(color="black", width=2),
                name="Daily probability"
            ))

            # X-axis ticks: show every day number
            tick_vals = date_range[indices]
            tick_text = [d.day for d in tick_vals]

            month_title = tick_vals[0].strftime("%B %Y")
            fig.update_layout(
                title=month_title,
                xaxis=dict(
                    tickmode="array",
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    tickangle=0,
                    title=None,
                ),
                yaxis_title="Simulation index / probability scale",
                height=350,
                showlegend=False,
                margin=dict(l=40, r=20, t=50, b=40),
            )

            figures.append(fig)

        # --------------------------------------------------
        # 3. Render ALL figures into ONE HTML file
        # --------------------------------------------------
        html_parts = [
            to_html(fig, full_html=False, include_plotlyjs="cdn")
            for fig in figures
        ]

        html = f"""
        <html>
        <head>
            <title>Period Prediction Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                }}
            </style>
        </head>
        <body>
            <h1>Period Prediction — Monthly View</h1>
            {''.join(html_parts)}
        </body>
        </html>
        """

        with open(output_file, "w") as f:
            f.write(html)

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
    parser.add_argument('--output', type=str, default=f'period_prediction_report_{datetime.now().month}_{datetime.now().year}.html',
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
    report_file = predictor.generate_html_report(date_range, simulations,probabilities, args.output)
    
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
