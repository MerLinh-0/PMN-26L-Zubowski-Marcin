import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import numpy as np


def load_data(path):
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    data = pd.read_csv(path, header=None, names=columns)
    return data


def main():
    data = load_data("iris.data")
    x = data.iloc[:, :-1].values # wszystkie wiersze i kolumny oprócz ostatniej

    le = LabelEncoder()
    y = le.fit_transform(data['species'])

    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(x)
    clusters = kmeans.labels_

    new_labels = np.zeros_like(clusters) # tablica zer
    for i in range(3):
        mask = (clusters == i) 
        chosen_labels = y[mask] 
        dominant_label = np.bincount(chosen_labels).argmax()
        new_labels[mask] = dominant_label

    print(new_labels)


main()
