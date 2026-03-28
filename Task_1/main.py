import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_data(path):
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    data = pd.read_csv(path, header=None, names=columns)
    return data


def print_metrics(y, new_labels):
    print("--- Metryki ---")
    print(f"Dokładność: {accuracy_score(y, new_labels):.4f}")
    print(f"Precyzja: {precision_score(y, new_labels, average='weighted'):.4f}")
    print(f"Czułość: {recall_score(y, new_labels, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y, new_labels, average='weighted'):.4f}")
    print(f"Macierz pomyłek:\n{confusion_matrix(y, new_labels)}")


def main():
    data = load_data("iris.data")
    x = data.iloc[:, :-1].values

    le = LabelEncoder()
    y = le.fit_transform(data['species'])

    # Algorytm k-means
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(x)
    clusters = kmeans.labels_ 

    # dopasowanie klas otrzymanych w k-means do rzeczywistych
    new_labels = np.zeros_like(clusters)
    for i in range(3):
        mask = (clusters == i) 
        chosen_labels = y[mask] 
        dominant_label = np.bincount(chosen_labels).argmax()
        new_labels[mask] = dominant_label

    print_metrics(y, new_labels)


main()
