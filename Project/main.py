from eda import load_data, analyze_data
from models import train_decision_tree, train_xgboost

# TODO:



def main():
    data, columns, data_encoded = load_data()
    # analyze_data(data, columns)

    print("\n=== Training Decision Tree Model ===")
    model_tree = train_decision_tree(data_encoded)

    print("\n=== Training XGBoost Model ===")
    model_xgb = train_xgboost(data_encoded)
    

main()
