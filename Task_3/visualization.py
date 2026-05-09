import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def plot_experiment_series_reults(results, param_name):
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    for param_val, history in results.items():
        plt.plot(history['test_loss'], marker='o', label=f'{param_name}: {param_val}')

    plt.title(f'Wpływ parametru {param_name} na test_loss (Błąd)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    for param_val, history in results.items():
        plt.plot(history['test_acc'], marker='o', label=f'{param_name}: {param_val}')

    plt.title(f'Wpływ parametru {param_name} na test_accuracy (Skuteczność)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def generate_results_table(results, param_name):
    data = []
    for param_val, history in results.items():
        final_loss = history['test_loss'][-1]
        final_accuracy = history['test_acc'][-1]
        best_accuracy = max(history['test_acc'])
        data.append({
            param_name: param_val, 
            'Final Test Loss': final_loss, 
            'Final Test Accuracy': final_accuracy, 
            'Best Test Accuracy': best_accuracy
        })

    df = pd.DataFrame(data)
    df.set_index(param_name, inplace=True)
    return df


def metrics(model, test_loader, device, classess):
    model.eval()
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print('=' * 50)
    print("RAPORT KLASYFIKACJI")
    print('=' * 50)
    print(classification_report(all_labels, all_predicted, target_names=classess))

    cm = confusion_matrix(all_labels, all_predicted)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Macierz Pomyłek - CIFAR-10')
    plt.show()


def show_misclassified_images(model, test_loader, device, classess, num_images=10):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified_images.append(inputs[i].cpu())
                    misclassified_labels.append(labels[i].cpu())
                    misclassified_preds.append(predicted[i].cpu())

                if len(misclassified_images) >= num_images:
                    break
            if len(misclassified_images) >= num_images:
                break

    plt.figure(figsize=(15, 5))
    for i in range(len(misclassified_images)):
        img = misclassified_images[i].permute(1, 2, 0).numpy()
        true_label = classess[misclassified_labels[i]]
        pred_label = classess[misclassified_preds[i]]
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f'True: {true_label}\nPred: {pred_label}')
        plt.axis('off')
    plt.suptitle('Przykłady Błędnie Klasyfikowanych Obrazów')
    plt.tight_layout()
    plt.show()