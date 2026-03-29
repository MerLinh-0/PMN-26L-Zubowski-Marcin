import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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


def visualize_clusters(x, labels):
    tsne = TSNE(n_components=2, random_state=0)
    x_2d = tsne.fit_transform(x)

    plt.scatter(x_2d[:, 0], x_2d[:, 1], c=labels, cmap='viridis')
    plt.title("Wizualizacja grupowania K-means za pomocą t-SNE")
    plt.xlabel("TSNE wymiar 1")
    plt.ylabel("TSNE wymiar 2")
    plt.colorbar()
    plt.show()


def main():
    data = load_data("iris.data")

    x = data.iloc[:, :-1].values
    le = LabelEncoder()
    y = le.fit_transform(data['species'])

    # algorytm k-means
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(x)
    labels = kmeans.labels_ 

    # dopasowanie klas otrzymanych w k-means do rzeczywistych
    new_labels = np.zeros_like(labels)
    for i in range(3):
        mask = (labels == i)
        chosen_labels = y[mask] 
        dominant_label = np.bincount(chosen_labels).argmax()
        new_labels[mask] = dominant_label

    # metryki i wizualizacja
    print_metrics(y, new_labels)
    visualize_clusters(x, new_labels)


main()
