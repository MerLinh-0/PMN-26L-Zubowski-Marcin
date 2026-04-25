import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression_model import train_logistic_regression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

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


def visualize_decision_boundary(X, y, features):
    model_2d, scaler_2d, X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_logistic_regression(X, y, selected_features=features)

    def plot_decision_boundary(model, X_scaled, y_true, title, ax):
        h = 0.05
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), cmap='coolwarm', alpha=0.8)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        plt.colorbar(contour, ax=ax, label='P(Sick)')

        for label, color, marker, name in [(0, 'blue', 'o', 'Healthy'), (1, 'red', 'X', 'Sick')]:
            mask = y_true == label
            ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=color, marker=marker,
                       edgecolors='k', linewidths=0.5, alpha=0.9, label=name, s=50)

        ax.set_xlabel(f"{features[0]}")
        ax.set_ylabel(f"{features[1]}")
        ax.set_title(title, fontsize=12)
        ax.legend(loc='upper right')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_decision_boundary(model_2d, X_train_2d, y_train_2d, "Decision Boundary on Training Set", axes[0])
    plot_decision_boundary(model_2d, X_test_2d, y_test_2d, "Decision Boundary on Test Set", axes[1])

    plt.suptitle(f"Logistic Regression Decision Boundary ({features[0]} vs {features[1]})", fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_raport(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Sick (1)']))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy (0)', 'Sick (1)'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    disp.plot(ax=axes[0], cmap='Blues')
    axes[0].set_title("Confusion Matrix")
    # ROC Curve
    y_scores = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'Logistic Regression (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier (AUC = 0.500)')
    axes[1].fill_between(fpr, tpr, alpha=0.1, color='blue')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate (Recall)')
    axes[1].set_title('ROC Curve', fontsize=13)
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print(f"\nAUC = {roc_auc:.3f}")

