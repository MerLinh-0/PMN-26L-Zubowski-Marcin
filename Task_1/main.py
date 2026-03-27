import pandas as pd


def load_data(path):
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    data = pd.read_csv(path, header=None, names=columns)
    return data


def main():
    data = load_data("iris.data")
    print(data.iloc[:, :-1].values) # wszystkie wiersze i kolumny oprócz ostatniej

    
main()
