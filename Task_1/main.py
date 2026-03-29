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


def visualize_clusters_TSNE(x, labels):
    tsne = TSNE(n_components=2, random_state=0)
    x_2d = tsne.fit_transform(x)

    scatter = plt.scatter(x_2d[:, 0], x_2d[:, 1], c=labels, cmap='viridis')
    plt.title("Wizualizacja grupowania K-means za pomocą t-SNE")
    plt.xlabel("TSNE wymiar 1")
    plt.ylabel("TSNE wymiar 2")
    species_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    plt.legend(handles=scatter.legend_elements()[0], labels=species_labels, title="Gatunki")
    plt.savefig("./images/wykres_tsne.png", dpi=300, bbox_inches='tight')
    plt.show()


def visualize_1d_clusters(x, labels, characteristic):
    scatter = plt.scatter(x, np.zeros_like(x), c=labels, cmap='viridis', alpha=0.6)
    plt.title(f"Wizualizacja grupowania K-means dla {characteristic}")
    plt.xlabel(f"Wartość {characteristic} [cm]")
    plt.yticks([])
    species_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    plt.legend(handles=scatter.legend_elements()[0], labels=species_labels, title="Gatunki")
    plt.savefig(f"./images/wykres_1d_{characteristic}.png", dpi=300, bbox_inches='tight')
    plt.show()


def clustering(x, y, title):
    print(f"\n===[ GRUPOWANIE DLA: {title} ]===")
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

    return new_labels


def main():
    data = load_data("iris.data")
    le = LabelEncoder()
    y = le.fit_transform(data['species'])

    # WSZYSTKIE CECHY
    x_all = data.iloc[:, :-1].values
    labels_all = clustering(x_all, y, "WSZYSTKIE CECHY")
    print_metrics(y, labels_all)
    visualize_clusters_TSNE(x_all, labels_all)

    # POJEDYNCZE CECHY
    characteristics = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for characteristic in characteristics:
        x_1d = data[[characteristic]].values
        labels_1d = clustering(x_1d, y, characteristic)
        print_metrics(y, labels_1d)
        visualize_1d_clusters(x_1d, labels_1d, characteristic)


main()
