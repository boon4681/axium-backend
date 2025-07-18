from axium.template import AxiumTemplate
import pandas as pd
import os

class ClassificationLoadCSVData(AxiumTemplate):
    name = "ClassificationLoadCSVData"
    id = "ClassificationLoadCSVData"
    category = "classification"

    input = {
        "file_path": "axium.str",
        "target_column": "axium.str",
        "separator": "axium.str",
        "encoding": "axium.str"
    }
    output = {
        "features": "axium.dataframe",
        "target": "axium.series"
    }

    @classmethod
    def run(cls, file_path, target_column, separator=',', encoding='utf-8'):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path, sep=separator, encoding=encoding)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        target = df[target_column].copy()
        features = df.drop(columns=[target_column])
        return {"features": features, "target": target}
