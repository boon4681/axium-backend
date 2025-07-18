from axium.template import AxiumTemplate
import pandas as pd
from sklearn.datasets import load_iris, load_wine, make_classification

class ClassificationLoadSampleDataset(AxiumTemplate):
    name = "ClassificationLoadSampleDataset"
    id = "ClassificationLoadSampleDataset"
    category = "classification"

    input = {
        "dataset_name": "axium.str"
    }
    output = {
        "features": "axium.dataframe",
        "target": "axium.series"
    }

    @classmethod
    def run(cls, dataset_name="iris"):
        if dataset_name == "iris":
            dataset = load_iris()
            feature_names = dataset.feature_names
        elif dataset_name == "wine":
            dataset = load_wine()
            feature_names = dataset.feature_names
        elif dataset_name == "synthetic":
            X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=42)
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            features = pd.DataFrame(X, columns=feature_names)
            target = pd.Series(y, name='target')
            return {"features": features, "target": target}
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from ['iris', 'wine', 'synthetic']")
        features = pd.DataFrame(dataset.data, columns=feature_names)
        target = pd.Series(dataset.target, name='target')
        return {"features": features, "target": target}
