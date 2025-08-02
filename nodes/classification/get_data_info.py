from axium.node_typing import AxiumNode

class ClassificationGetDataInfo(AxiumNode):
    id = "classification.get-data-info"
    category = "classification"
    name = "Get Data Info"

    inputs = {
        "data": ("pandas.df", {})
    }
    outputs = {
        "info": ("dict", {})
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
        data = inputs.get("data")
        target_col = parameters.get("target") if parameters else None
        if data is None:
            return {"info": {"message": "No data loaded"}}
        features = data.drop(columns=[target_col]) if target_col and target_col in data.columns else data
        target = data[target_col] if target_col and target_col in data.columns else None
        info = {
            "shape": features.shape,
            "features": list(features.columns),
            "feature_types": dict(features.dtypes),
            "missing_values": dict(features.isnull().sum()),
        }
        if target is not None:
            info["target_stats"] = {
                "min": float(target.min()),
                "max": float(target.max()),
                "mean": float(target.mean()),
                "std": float(target.std()),
                "missing": int(target.isnull().sum())
            }
        return {"info": info}
