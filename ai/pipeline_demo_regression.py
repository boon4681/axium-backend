from pipelines.tabular.regression.regression_01_load_data import RegressionDataLoader


def demo_regression():
    print("--- Regression Data Loader Demo ---")
    reg_loader = RegressionDataLoader()
    X, y = reg_loader.load_sample_dataset("california_housing")
    print(f"Regression data shape: {X.shape}, Target shape: {y.shape}")
    print(f"Regression data info: {reg_loader.get_data_info()}")


def main():
    demo_regression()
    print()


if __name__ == "__main__":
    main()
