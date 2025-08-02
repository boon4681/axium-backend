from axium.node_typing import AxiumNode

class RegressionAnalyzeData(AxiumNode):
    id = "regression.analyze-data"
    category = "regression"
    name = "Analyze Data"

    inputs = {
        "data": ("pandas.df", {})
    }
    outputs = {
        "analysis": ("dict", {})
    }
    parameters = None

    @classmethod
    def validate_inputs(cls, inputs: dict):
        data = inputs.get("data")
        if data is None:
            return {"error": "Data is required"}
        return {}

    @classmethod
    def run(cls, parameters: dict, inputs: dict):
        import numpy as np
        data = inputs.get("data")
        target_col = parameters.get("target") if parameters else None
        
        if data is None:
            return {"analysis": None}
        
        if target_col and target_col in data.columns:
            features = data.drop(columns=[target_col])
            target = data[target_col]
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
        else:
            analysis = {
                'shape': data.shape,
                'numeric_features': list(data.select_dtypes(include=[np.number]).columns),
                'categorical_features': list(data.select_dtypes(include=['object', 'category']).columns),
                'missing_values': dict(data.isnull().sum()),
            }
        return {"analysis": analysis}
