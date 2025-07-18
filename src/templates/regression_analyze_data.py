from axium.template import AxiumTemplate
import pandas as pd
import numpy as np

class RegressionAnalyzeData(AxiumTemplate):
    name = "RegressionAnalyzeData"
    id = "RegressionAnalyzeData"
    category = "regression"

    input = {
        "features": "axium.dataframe",
        "target": "axium.series"
    }
    output = {
        "analysis": "axium.dict"
    }

    @classmethod
    def run(cls, features, target):
        analysis = {
            'shape': features.shape,
            'numeric_features': list(features.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(features.select_dtypes(include=['object', 'category']).columns),
            'missing_values': dict(features.isnull().sum()),
            'target_stats': {
                'mean': float(target.mean()),
                'std': float(target.std()),
                'min': float(target.min()),
                'max': float(target.max()),
                'median': float(target.median()),
                'skewness': float(target.skew()),
                'kurtosis': float(target.kurtosis())
            },
            'feature_correlations': features.select_dtypes(include=[np.number]).corrwith(target).abs().sort_values(ascending=False).to_dict()
        }
        return {"analysis": analysis}
