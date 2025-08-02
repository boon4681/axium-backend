from axium.template import AxiumTemplate

class ClassificationGetDataInfo(AxiumTemplate):
    name = "ClassificationGetDataInfo"
    id = "ClassificationGetDataInfo"
    category = "classification"

    input = {
        "features": "axium.dataframe",
        "target": "axium.series"
    }
    output = {
        "info": "axium.dict"
    }

    @classmethod
    def run(cls, features, target):
        if features is None:
            return {"info": {"message": "No data loaded"}}
        return {
            "info": {
                "shape": features.shape,
                "features": list(features.columns),
                "feature_types": dict(features.dtypes),
                "missing_values": dict(features.isnull().sum()),
                "target_stats": {
                    "min": float(target.min()),
                    "max": float(target.max()),
                    "mean": float(target.mean()),
                    "std": float(target.std()),
                    "missing": int(target.isnull().sum())
                }
            }
        }
