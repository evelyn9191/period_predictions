# Period Prediction Tool

A Python-based tool for predicting future menstrual cycles and visualizing 
period probabilities using historical cycle data. Ideal if you're planning 
vacations or other events long time ahead.

## Features

- Predicts future period start dates based on historical cycle data
- Generates interactive visualizations of period probabilities

## Prerequisites

- Python 3.7+

## Installation

Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your cycle data in a CSV file named `cycle_data.csv` with the following format:
   ```
   start_date,end_date,cycle_length
   YYYY-MM-DD,YYYY-MM-DD,28
   YYYY-MM-DD,YYYY-MM-DD,27
   ...
   ```
- `start_date`: The first day of the period (YYYY-MM-DD)
- `end_date`: The last day of the period (YYYY-MM-DD)
- `cycle_length`: Length of the cycle in days

2. Run the prediction script:
   ```bash
   python period_prediction.py
   ```

3. The script will generate an interactive HTML visualization showing predicted period probabilities
under `period_prediction_report.html`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.