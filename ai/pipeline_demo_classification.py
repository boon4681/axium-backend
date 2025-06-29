from pipelines.tabular.classification.classification_01_load_data import ClassificationDataLoader


def demo_classification():
    print("--- Classification Data Loader Demo ---")
    clf_loader = ClassificationDataLoader()
    X, y = clf_loader.load_sample_dataset("iris")
    print(f"Classification data shape: {X.shape}, Target shape: {y.shape}")
    print(f"Classification data info: {clf_loader.get_data_info()}")


def main():
    demo_classification()


if __name__ == "__main__":
    main()
