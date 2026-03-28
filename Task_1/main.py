import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


def load_data(path):
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    data = pd.read_csv(path, header=None, names=columns)
    return data


def main():
    data = load_data("iris.data")
    x = data.iloc[:, :-1].values # wszystkie wiersze i kolumny oprócz ostatniej

    le = LabelEncoder()
    y = le.fit_transform(data['species'])

    print(x)
    print(y)
    
main()
