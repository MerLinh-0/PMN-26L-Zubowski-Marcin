from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from visualization import visualize_categorical_feature_distributions, visualize_numeric_feature_distributions, visualize_binary_feature_distributions, visualize_target_distribution


def load_data():
    data_path = Path(__file__).with_name("banking_transactions.csv")
    data = pd.read_csv(data_path)
    if 'transaction_id' in data.columns:
        data = data.drop('transaction_id', axis=1) # transaction_id to unikalny identyfikator i nie wnosi wartości do analizy

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'str']).columns.tolist()
    binary_columns = [col for col in numeric_columns if data[col].nunique() == 2]
    numeric_columns = [col for col in numeric_columns if col not in binary_columns]

    features = {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "binary": binary_columns
    }

    # Zamiana danych kategorycznych przez One-Hot Encoding
    data_encoded = pd.get_dummies(data, columns=categorical_columns, dtype=int)
    data_encoded['fraud_flag'] = data_encoded['fraud_flag'].astype(int)

    return data, features, data_encoded


def analyze_data(data, features):

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

    print("\n- Numeric Feature Distributions -")
    for col in features["numeric"]:
        print(f"{col}: min={data[col].min()}, max={data[col].max()}, mean={data[col].mean():.2f}, median={data[col].median():.2f}")
    visualize_numeric_feature_distributions(data, features["numeric"])

    print("\n- Categorical Feature Distributions -")
    for col in features["categorical"]:
        print(f"{col}: {data[col].value_counts().to_dict()}")
    visualize_categorical_feature_distributions(data, features["categorical"])

    print("\n- Binary Feature Distributions -")
    for col in features["binary"]:
        print(f"{col}: {data[col].value_counts().to_dict()}")
    visualize_binary_feature_distributions(data, features["binary"])

    print("\n- Target Distribution -")
    class_counts = data['fraud_flag'].value_counts()
    print(class_counts)
    visualize_target_distribution(data)


    # TODO: Wykresy np. korelacje, transaction_amount VS fraud_flag, 
    # geo_distance_km VS device_risk_score (sprawdzić czy wpływa na device_risk_score) itp.
