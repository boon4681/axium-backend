from axium.node_typing import AxiumNode

class LoadCSV(AxiumNode):
    id = "base.load-csv"
    category = "base"
    name = "Load CSV"

    outputs = {
        "result": ("pandas.df", {})
    }
    parameters = {
        "file": ("axium.file", {
            # file_type is glob format
            "file_type": "csv",
            "placeholder": "Select file .csv",
            "label": "File",
            "inline": False
        })
    }

    def run(parameters: dict, inputs: dict):
        return {}
