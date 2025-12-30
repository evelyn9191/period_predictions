# Period Prediction Tool

> [!NOTE]
> **This code was written by SWE-1, Claude Sonnet 4.5, and GPT-5** 🤖

A Python-based tool for predicting future menstrual cycles using Monte Carlo simulations. 
Unlike simple period tracking apps that predict specific dates and are not very accurate 
especially when it comes to long-term predictions, this tool shows **probability distributions** for each day, 
accounting for natural cycle variability. Perfect for vacation planning and long-term scheduling.

## Key Features

- **Monte Carlo Simulation**: Runs 10,000+ simulations to model cycle variability
- **Probability-Based Predictions**: Shows likelihood of having your period on each day (not just single dates)
- **12-Month Forecast**: Generates predictions for the next year
- **Interactive HTML Report**: Hover over dates to see exact probabilities

## How It Works

The algorithm uses your historical cycle data to:
1. Calculate the distribution of your cycle lengths (mean ± standard deviation)
2. Model period duration variability
3. Run thousands of simulations with random sampling from these distributions
4. Aggregate results to show the probability of having your period on each future day

**Key Insight**: Predictions become "fuzzier" further into the future because cycle variability compounds over time. This is realistic and helps you identify truly safe dates vs. risky ones.

## Prerequisites

- Python 3.9+

## Installation

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your cycle data in a CSV file named `cycle_data.csv` (see [cycle_data_sample.csv](cycle_data_sample.csv) for an example). 
You can e.g. printscreen your menstrual calendar app and ask AI to convert it into a CSV file with the following format:
   ```
   start_date,end_date,cycle_length,period_duration
   2025-11-13,2025-12-11,29,7
   2025-10-19,2025-11-12,25,6
   ...
   ```
   - `start_date`: First day of your period (YYYY-MM-DD)
   - `end_date`: Date before next period starts (YYYY-MM-DD)
   - `cycle_length`: Days from this period start to next period start
   - `period_duration`: Number of days your period lasted (bleeding days)

2. Run the prediction script:
   ```bash
   python period_prediction.py
   ```

3. Open the generated `period_prediction_report.html` in your browser

## Interpreting Results

- **Probability < 10%**: Very unlikely - safe for planning
- **Probability 10-30%**: Low risk - generally safe
- **Probability 30-50%**: Moderate risk - consider backup plans
- **Probability 50-70%**: High risk - likely to have period
- **Probability > 70%**: Very high risk - avoid if possible

## Advanced Options

```bash
python period_prediction.py --csv your_data.csv --output report.html --simulations 20000
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.