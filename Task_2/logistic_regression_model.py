from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_logistic_regression(X, y, selected_features=None):
    if selected_features is not None:
        X = X[selected_features].values
    else:
        X = X.values

    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled) 
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.2f}")
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test
