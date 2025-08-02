from axium.node_typing import AxiumNode

class ClassificationLoadSampleDataset(AxiumNode):
    id = "classification.load-sample-dataset"
    category = "classification"
    name = "Load Sample Dataset"

    outputs = {
        "data": ("pandas.df", {})
    }
    parameters = None

    @classmethod
    def validate_inputs(cls, inputs: dict):
        return {}

    @classmethod
    def run(cls, parameters: dict, inputs: dict):
        import pandas as pd
        from sklearn.datasets import load_iris, load_wine, make_classification
        dataset_name = parameters.get("dataset_name", "iris") if parameters else "iris"
        if dataset_name == "iris":
            dataset = load_iris()
            feature_names = dataset.feature_names
            features = pd.DataFrame(dataset.data, columns=feature_names)
            target = pd.Series(dataset.target, name='target')
        elif dataset_name == "wine":
            dataset = load_wine()
            feature_names = dataset.feature_names
            features = pd.DataFrame(dataset.data, columns=feature_names)
            target = pd.Series(dataset.target, name='target')
        elif dataset_name == "synthetic":
            X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=42)
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            features = pd.DataFrame(X, columns=feature_names)
            target = pd.Series(y, name='target')
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from ['iris', 'wine', 'synthetic']")
        return {"features": features, "target": target}
