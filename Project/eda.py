import pandas as pd
import matplotlib.pyplot as plt
from visualization import visualize_feature_distributions, visualize_target_distribution


def load_data():
    data = pd.read_csv('banking_transactions.csv')

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'str']).columns.tolist()

    if 'transaction_id' in numeric_columns:
        numeric_columns.remove('transaction_id')

    return data, numeric_columns, categorical_columns


def analyze_data():
    data, numeric_columns, categorical_columns = load_data()
    print("== EXPLORATORY DATA ANALYSIS ==")
    print(data.head())
    
    print("\n- Dataset Shape -")
    print(data.shape)

    print("\n- Dataset Information -")
    print(data.info())

    print("\n- Missing Values -")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])
    print(f"{missing_values.sum()} missing values")

    print("\n- Feature Distributions -")
    for col in numeric_columns:
        print(f"{col}: min={data[col].min()}, max={data[col].max()}, mean={data[col].mean():.2f}, median={data[col].median():.2f}")
    visualize_feature_distributions(data, numeric_columns)

    print("\n- Target Distribution -")
    class_counts = data['fraud_flag'].value_counts()
    print(class_counts)
    visualize_target_distribution(data)

    # TODO: Wykresy np. korelacje, transaction_amount VS fraud_flag, 
    # geo_distance_km VS device_risk_score (sprawdzić czy wpływa na device_risk_score) itp.


analyze_data()
