from eda import load_data, analyze_data
from models import train_decision_tree, train_xgboost

# TODO:
# Optymalizacja parametrów modeli
# Uzupełnienie raportu
# Więcej wizualizacji



def main():
    data, columns, data_encoded = load_data()
    # analyze_data(data, columns)

    print("\n=== Training Decision Tree Model ===")
    model = train_decision_tree(data_encoded)

    print("\n=== Training XGBoost Model ===")
    model = train_xgboost(data_encoded)
    

main()
