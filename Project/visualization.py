import matplotlib.pyplot as plt

def visualize_target_distribution(data):
    class_counts = data['fraud_flag'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Legal (False)', 'Fraud (True)'], class_counts.values, color=['mediumseagreen', 'tomato'], alpha=0.7)
    ax.set_title('Distribution of Target Variable', fontweight='bold')
    ax.set_ylabel('Transaction Count')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def visualize_feature_distributions(data, numeric_columns):
    num_rows = (len(numeric_columns) + 3) // 4
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    for i, col in enumerate(numeric_columns):
        if col == "transaction_time_hour" or col == "failed_transactions_last_30d":
            axes[i].hist(data[col], bins=range(0, 25), edgecolor='k', alpha=0.7)
        else:
            axes[i].hist(data[col], bins=30, edgecolor='k', alpha=0.7)
        axes[i].set_title(f'{col}')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

