from axium.node_typing import AxiumNode

class ClassificationAnalyzeData(AxiumNode):
    id = "classification.analyze-data"
    category = "classification"
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
        if data is None:
            return {"analysis": None}
        analysis = {
            'shape': data.shape,
            'numeric_features': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(data.select_dtypes(include=['object', 'category']).columns),
            'missing_values': dict(data.isnull().sum()),
        }
        # If target column exists, add target analysis
        target_col = parameters.get("target") if parameters else None
        if target_col and target_col in data.columns:
            target = data[target_col]
            analysis.update({
                'target_classes': list(target.unique()),
                'target_distribution': dict(target.value_counts()),
                'class_balance': target.value_counts(normalize=True).to_dict()
            })
        return {"analysis": analysis}
