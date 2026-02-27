import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

class BenchmarkModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, data):
        """
        Fit a linear trend to the benchmark scores over time.
        data: DataFrame with 'Date' and 'Score' columns
        """
        # Convert date to ordinal (number of days)
        # Ensure Date is datetime
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data['DateOrdinal'] = data['Date'].apply(lambda x: x.toordinal()).values.reshape(-1, 1)

        X = data[['DateOrdinal']]
        y = data['Score']

        self.model.fit(X, y)

    def predict(self, date):
        """
        Predict the benchmark score for a specific date.
        date: string 'YYYY-MM-DD' or datetime object
        """
        if not hasattr(self.model, 'coef_'):
             raise ValueError("Model has not been trained yet.")

        date_ordinal = pd.to_datetime(date).toordinal()
        return self.model.predict([[date_ordinal]])[0]

class CadenceModel:
    def __init__(self):
        pass

    def predict_next_release(self, data):
        """
        Predict the next release date based on average time between releases.
        data: DataFrame with 'Date' column
        """
        if len(data) < 2:
            return None # Not enough data to predict cadence

        # Ensure Date is datetime and sort
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')

        data['TimeDelta'] = data['Date'].diff().dt.days

        avg_days = data['TimeDelta'].dropna().mean()
        last_release = data['Date'].iloc[-1]

        next_release_date = last_release + pd.Timedelta(days=avg_days)
        return next_release_date
