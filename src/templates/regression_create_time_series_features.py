from axium.template import AxiumTemplate
import pandas as pd
import numpy as np

class RegressionCreateTimeSeriesFeatures(AxiumTemplate):
    name = "RegressionCreateTimeSeriesFeatures"
    id = "RegressionCreateTimeSeriesFeatures"
    category = "regression"

    input = {
        "df": "axium.dataframe",
        "date_column": "axium.str",
        "lag_features": "axium.int"
    }
    output = {
        "result": "axium.dataframe"
    }

    @classmethod
    def run(cls, df, date_column, lag_features=5):
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found")
        df_ts = df.copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        df_ts = df_ts.sort_values(date_column)
        df_ts['year'] = df_ts[date_column].dt.year
        df_ts['month'] = df_ts[date_column].dt.month
        df_ts['day'] = df_ts[date_column].dt.day
        df_ts['dayofweek'] = df_ts[date_column].dt.dayofweek
        df_ts['quarter'] = df_ts[date_column].dt.quarter
        df_ts['is_weekend'] = (df_ts[date_column].dt.dayofweek >= 5).astype(int)
        numeric_cols = df_ts.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend']]
        for col in numeric_cols:
            for lag in range(1, lag_features + 1):
                df_ts[f'{col}_lag_{lag}'] = df_ts[col].shift(lag)
        for col in numeric_cols:
            df_ts[f'{col}_rolling_mean_3'] = df_ts[col].rolling(window=3).mean()
            df_ts[f'{col}_rolling_std_3'] = df_ts[col].rolling(window=3).std()
        df_ts = df_ts.dropna()
        return {"result": df_ts}
