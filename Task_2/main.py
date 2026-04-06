import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
  

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


def visualize_features(X, y):
    df = pd.concat([X, y], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    df.hist(ax=ax, bins=15, edgecolor='black', color='skyblue', grid=False)
    plt.suptitle('Distribution of features in Heart Disease dataset', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    

def main():
    X, y = get_clean_data()
    print(f"Dataset shape: {X.shape}, Target distribution:\n{y.value_counts()}")
    visualize_features(X, y)


main()
