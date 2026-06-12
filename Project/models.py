from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from xgboost import XGBClassifier, plot_importance

import matplotlib.pyplot as plt


def train_decision_tree(data):
    X = data.drop('fraud_flag', axis=1)
    y = data['fraud_flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt = tree.DecisionTreeClassifier(class_weight='balanced', random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    print("\n=== Decision Tree Model Performance ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(20, 10))
    tree.plot_tree(dt, filled=True, feature_names=X.columns, class_names=['Legal', 'Fraud'], rounded=True)
    plt.title("Decision Tree Visualization", fontsize=16, fontweight='bold')
    plt.show()

    return dt


def train_xgboost(data):
    X = data.drop('fraud_flag', axis=1)
    y = data['fraud_flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    legal_count = sum(y == 0)
    fraud_count = sum(y == 1)
    weight = legal_count / fraud_count

    xgb = XGBClassifier(scale_pos_weight=weight, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    print("\n=== XGBoost Model Performance ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    plot_importance(xgb, max_num_features=10, height=0.5, title='Top 10 Feature Importance')
    plt.show()

    return xgb