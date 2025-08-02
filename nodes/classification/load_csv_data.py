from axium.node_typing import AxiumNode

class ClassificationLoadCSVData(AxiumNode):
    id = "classification.load-csv-data"
    category = "classification"
    name = "Load CSV Data"

    outputs = {
        "data": ("pandas.df", {})
    }
    parameters = {
        "file": ("axium.file", {
            "file_type": "csv",
            "placeholder": "Select file .csv",
            "label": "File",
            "inline": False
        })
    }

    @classmethod
    def validate_inputs(cls, inputs: dict):
        return {}

    @classmethod
    def run(cls, parameters: dict, inputs: dict):
        import pandas as pd
        import os
        file_path = parameters.get("file") if parameters else None
        target_column = parameters.get("target_column") if parameters else None
        separator = parameters.get("separator", ",") if parameters else ","
        encoding = parameters.get("encoding", "utf-8") if parameters else "utf-8"
        if not file_path or not os.path.exists(file_path):
            return {"data": None}
        df = pd.read_csv(file_path, sep=separator, encoding=encoding)
        if target_column and target_column in df.columns:
            target = df[target_column].copy()
            features = df.drop(columns=[target_column])
            data = df
        else:
            target = None
            features = df
            data = df
        return {"data": data}
