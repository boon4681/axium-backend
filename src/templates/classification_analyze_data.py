from axium.template import AxiumTemplate
import pandas as pd
import numpy as np

class ClassificationAnalyzeData(AxiumTemplate):
    name = "ClassificationAnalyzeData"
    id = "ClassificationAnalyzeData"
    category = "classification"

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
            'target_classes': list(target.unique()),
            'target_distribution': dict(target.value_counts()),
            'class_balance': target.value_counts(normalize=True).to_dict()
        }
        return {"analysis": analysis}
