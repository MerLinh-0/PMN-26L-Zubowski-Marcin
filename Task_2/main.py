import pandas as pd
from ucimlrepo import fetch_ucirepo 
from logistic_regression_model import train_logistic_regression
from visualization import visualize_features, visualize_weights

# Data loading and cleaning to prepare the dataset for analysis and modeling
def get_clean_data():
    heart_disease = fetch_ucirepo(id=45)
    X_raw = heart_disease.data.features
    y_raw = heart_disease.data.targets
    # df.info()

    df = pd.concat([X_raw, y_raw], axis=1)
    # print(df[df.isnull().any(axis=1)]) # Check for missing values
    df_cleaned = df.dropna()
    df_cleaned["num"] = df_cleaned["num"].apply(lambda x: 1 if x > 0 else 0) # Convert to binary classification

    X = df_cleaned.drop(columns=["num"])
    y = df_cleaned["num"]

    return X, y


def statistics(X):
    for col in X.columns:
        print(f"{col} - Mean: {X[col].mean():.2f}, Median: {X[col].median():.2f}, Min: {X[col].min():.2f}, Max: {X[col].max():.2f}")


def main():
    X, y = get_clean_data()
    print(f"Dataset shape: {X.shape}, Target distribution:\n{y.value_counts()}")
    # visualize_features(X, y)
    # statistics(X)
    model = train_logistic_regression(X, y)
    visualize_weights(model, X)

main()
