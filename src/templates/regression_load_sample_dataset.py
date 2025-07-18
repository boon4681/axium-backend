from axium.template import AxiumTemplate
import pandas as pd
from sklearn.datasets import load_diabetes, make_regression, fetch_california_housing

class RegressionLoadSampleDataset(AxiumTemplate):
    name = "RegressionLoadSampleDataset"
    id = "RegressionLoadSampleDataset"
    category = "regression"

    input = {
        "dataset_name": "axium.str"
    }
    output = {
        "features": "axium.dataframe",
        "target": "axium.series"
    }

    @classmethod
    def run(cls, dataset_name="california_housing"):
        if dataset_name == "california_housing":
            dataset = fetch_california_housing()
            feature_names = dataset.feature_names
        elif dataset_name == "diabetes":
            dataset = load_diabetes()
            feature_names = dataset.feature_names
        elif dataset_name == "synthetic":
            X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, noise=0.1, random_state=42)
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            features = pd.DataFrame(X, columns=feature_names)
            target = pd.Series(y, name='target')
            return {"features": features, "target": target}
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from ['california_housing', 'diabetes', 'synthetic']")
        features = pd.DataFrame(dataset.data, columns=feature_names)
        target = pd.Series(dataset.target, name='target')
        return {"features": features, "target": target}
