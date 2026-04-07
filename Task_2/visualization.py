import pandas as pd
import matplotlib.pyplot as plt

# Basic visualization for each feature in the dataset (histograms)
def visualize_features(X, y):
    df = pd.concat([X, y], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    df.hist(ax=ax, bins=15, edgecolor='black', color='skyblue', grid=False)
    plt.suptitle('Distribution of features in Heart Disease dataset', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_weights(model, X):
    coef_df = pd.DataFrame(
        {"Feature": X.columns, "Weight": model.coef_[0]}
    ).sort_values("Weight")

    plt.figure(figsize=(10, 8))
    colors = ['red' if w < 0 else 'steelblue' for w in coef_df['Weight']]
    plt.barh(coef_df['Feature'], coef_df['Weight'], color=colors)
    plt.xlabel('Weight (coefficient)')
    plt.title('Logistic Regression — Learned Feature Weights')
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.show()
