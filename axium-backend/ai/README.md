CLASSIFICATION PREPROCESSING OPTIONS:

# Basic usage

result = preprocessor.full_preprocessing_pipeline(X, y)

# Custom missing value handling

result = preprocessor.full_preprocessing_pipeline(
X, y,
missing_value_strategy='median', # 'mean', 'median', 'mode', 'knn'
categorical_missing_strategy='constant' # 'most_frequent', 'constant'
)

# Custom scaling

result = preprocessor.full_preprocessing_pipeline(
X, y,
scaling_method='robust' # 'standard', 'minmax', 'robust'
)

# Feature selection

result = preprocessor.full_preprocessing_pipeline(
X, y,
select_features=True,
feature_selection_method='chi2', # 'chi2', 'f_classif'
feature_selection_k=15
)

# Custom categorical encoding

result = preprocessor.full_preprocessing_pipeline(
X, y,
categorical_encoding='ordinal', # 'onehot', 'ordinal', 'label'
max_categories=5
)

REGRESSION PREPROCESSING OPTIONS:

# With outlier removal

result = preprocessor.full_preprocessing_pipeline(
X, y,
remove_outliers=True,
outlier_method='iqr', # 'iqr', 'zscore'
outlier_threshold=2.0
)

# With target transformation

result = preprocessor.full_preprocessing_pipeline(
X, y,
transform_target=True,
target_transform_method='log' # 'log', 'sqrt', 'box-cox', 'yeo-johnson'
)

# With polynomial features

result = preprocessor.full_preprocessing_pipeline(
X, y,
create_polynomial=True,
poly_degree=2
)

# With advanced feature selection

result = preprocessor.full_preprocessing_pipeline(
X, y,
select_features=True,
feature_selection_method='mutual_info', # 'f_regression', 'mutual_info'
feature_selection_k=10
)
""")

def main():
"""Main demo function"""
print("IMPROVED PREPROCESSING WITH KWARGS DEMO")

    # Demo regression
    reg_results = demo_regression_with_kwargs()

    # Show usage examples
    show_usage_examples()

    print("\n" + "="*70)
    print("PREPROCESSING WITH KWARGS COMPLETED!")
    print("="*70)
    print("""

# Load your data

# X, y = your_data_loading_function()

# Classification

clf_preprocessor = ClassificationPreprocessor()
result = clf_preprocessor.full_preprocessing_pipeline(
X, y,
missing_value_strategy='median',
scaling_method='robust',
feature_selection_method='chi2',
feature_selection_k=10,
select_features=True
)

# Regression

reg_preprocessor = RegressionPreprocessor()
result = reg_preprocessor.full_preprocessing_pipeline(
X, y,
remove_outliers=True,
transform_target=True,
target_transform_method='log',
scaling_method='standard'
)

# Use the processed data

X_train, y_train = result['X_train'], result['y_train']
X_test, y_test = result['X_test'], result['y_test']
""")
