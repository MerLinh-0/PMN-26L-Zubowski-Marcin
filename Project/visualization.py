import matplotlib.pyplot as plt


def visualize_target_distribution(data):
    class_counts = data['fraud_flag'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Legal (False)', 'Fraud (True)'], class_counts.values, color=['mediumseagreen', 'tomato'], alpha=0.7)
    ax.set_title('Distribution of Target Variable', fontweight='bold')
    ax.set_ylabel('Transaction Count')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def visualize_numeric_feature_distributions(data, numeric_columns):
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


def visualize_categorical_feature_distributions(data, categorical_columns):
    fig, axes = plt.subplots(1, 2, figsize=(16, 16))
    axes.flatten()
    for i, col in enumerate(categorical_columns):

        counts = data[col].value_counts()
        bars = axes[i].bar(counts.index, counts.values, color='steelblue', alpha=0.7)
        axes[i].set_title(f'Category {col}', fontweight='bold')
        axes[i].set_ylabel('Amount of transactions')
        axes[i].tick_params(axis='x', rotation=45)

        for label in axes[i].get_xticklabels():
            label.set_horizontalalignment('right')

        for bar in bars:
            yval = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2, yval + (yval * 0.02), int(yval), ha='center', va='bottom', fontsize=10)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
    plt.subplots_adjust(bottom=0.20, hspace=0.5)
    plt.show()


def visualize_binary_feature_distributions(data, binary_columns):
    num_rows = (len(binary_columns) + 3) // 4
    fig, axes = plt.subplots(num_rows, 3, figsize=(20, 5 * num_rows))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    for i, col in enumerate(binary_columns):
        class_counts = data[col].value_counts()
        axes[i].pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=['lightskyblue', 'lightcoral'], startangle=90)
        axes[i].set_title(f'{col}', fontweight='bold')
    plt.show()
