import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class NeuralNetwork:
    def __init__(self, layers=[4, 32, 16, 8, 1], learning_rate=0.1):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        for i in range(len(layers) - 1):
            limit = np.sqrt(2 / layers[i])
            self.weights.append(np.random.randn(layers[i + 1], layers[i]) * limit)
            self.biases.append(np.zeros((layers[i + 1], 1)))

    def leaky_relu(self, x):
        return np.where(x > 0, x, 0.01 * x)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def forward(self, X):
        self.activations = [X.T]
        self.z_values = []

        current_activation = X.T

        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], current_activation) + self.biases[i]
            self.z_values.append(z)
            current_activation = self.leaky_relu(z)
            self.activations.append(current_activation)

        z = np.dot(self.weights[-1], current_activation) + self.biases[-1]
        self.z_values.append(z)
        output = self.sigmoid(z)
        self.activations.append(output)

        return output

    def backward(self, X, y, output):
        m = X.shape[0]
        y = y.reshape(1, -1)

        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        dZ = output - y
        dW[-1] = np.dot(dZ, self.activations[-2].T) / m
        db[-1] = np.sum(dZ, axis=1, keepdims=True) / m

        for l in range(len(self.weights) - 2, -1, -1):
            dA = np.dot(self.weights[l + 1].T, dZ)
            dZ = dA * self.leaky_relu_derivative(self.z_values[l])
            dW[l] = np.dot(dZ, self.activations[l].T) / m
            db[l] = np.sum(dZ, axis=1, keepdims=True) / m

        for l in range(len(self.weights)):
            self.weights[l] -= self.learning_rate * dW[l]
            self.biases[l] -= self.learning_rate * db[l]

        loss = np.mean((output - y) ** 2)
        return loss

    def train(self, X, y, epochs=2000, batch_size=64, validation_split=0.2, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y

        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        for epoch in range(epochs):
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            batch_count = 0

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.forward(X_batch)
                loss = self.backward(X_batch, y_batch, output)
                epoch_loss += loss
                batch_count += 1

            avg_train_loss = epoch_loss / batch_count

            val_output = self.forward(X_val)
            val_loss = np.mean((val_output - y_val.reshape(1, -1)) ** 2)

            train_preds = (self.forward(X_train) > 0.5).astype(int)
            train_acc = np.mean(train_preds == y_train.reshape(1, -1))

            val_preds = (val_output > 0.5).astype(int)
            val_acc = np.mean(val_preds == y_val.reshape(1, -1))

            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)

            if epoch % 200 == 0:
                print(f"Epoch {epoch:4d} | Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | Train Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return output.flatten()

    def predict_proba(self, X):
        return self.forward(X).flatten()

    def predict_class(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)


class SimpleNeuralNetwork:
    def __init__(self, input_size=4, hidden_size=16, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        self.w1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))

        self.w2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def forward(self, X):
        self.X = X.T

        self.z1 = np.dot(self.w1, self.X) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def compute_loss(self, y):
        return np.mean((self.a2 - y.reshape(1, -1)) ** 2)

    def backward(self, y, learning_rate=0.01):
        m = y.shape[0]
        y = y.reshape(1, -1)

        dZ2 = self.a2 - y
        dW2 = np.dot(dZ2, self.a1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.dot(self.w2.T, dZ2)
        dZ1 = dA1 * (self.z1 > 0)
        dW1 = np.dot(dZ1, self.X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.w1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.w2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs=200, batch_size=32, learning_rate=0.01, validation_split=0.2, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y

        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        for epoch in range(epochs):
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            batch_count = 0

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                self.forward(X_batch)
                loss = self.compute_loss(y_batch)
                self.backward(y_batch, learning_rate)

                epoch_loss += loss
                batch_count += 1

            avg_train_loss = epoch_loss / batch_count

            val_output = self.forward(X_val)
            val_loss = self.compute_loss(y_val)

            train_preds = (self.forward(X_train) > 0.5).astype(int)
            train_acc = np.mean(train_preds == y_train.reshape(1, -1))

            val_preds = (val_output > 0.5).astype(int)
            val_acc = np.mean(val_preds == y_val.reshape(1, -1))

            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | Train Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return output.flatten()

    def predict_proba(self, X):
        return self.forward(X).flatten()

    def predict_class(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)


def load_simple_model(filename='student_dropout_model.pkl'):
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)

    model = SimpleNeuralNetwork(
        input_size=model_data['input_size'],
        hidden_size=model_data['hidden_size'],
        output_size=model_data['output_size']
    )
    model.w1 = model_data['w1']
    model.b1 = model_data['b1']
    model.w2 = model_data['w2']
    model.b2 = model_data['b2']
    print(f"Simple model loaded from {filename}")
    return model


def load_normalization_params(filename='normalization_params.pkl'):
    with open(filename, 'rb') as f:
        norm_params = pickle.load(f)
    return norm_params['X_mean'], norm_params['X_std']


def plot_training_history(models_dict, model_names=None):
    if model_names is None:
        model_names = list(models_dict.keys())

    n_models = len(models_dict)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 2, 1], height_ratios=[1, 1])

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2])
    ]

    fig.suptitle('Сравнение моделей: История обучения', fontsize=16, fontweight='bold')

    colors = [
        '#E41A1C',
        '#377EB8',
        '#4DAF4A',
        '#984EA3',
        '#FF7F00',
        '#FFFF33',
        '#A65628',
        '#F781BF'
    ]

    color_names = [
        "Красный", "Синий", "Зеленый", "Фиолетовый",
        "Оранжевый", "Желтый", "Коричневый", "Розовый"
    ]

    legend_handles_upper = []
    legend_labels_upper = []

    legend_handles_lower = []
    legend_labels_lower = []

    for idx, (name, model) in enumerate(models_dict.items()):
        color_train_loss = colors[(idx * 4) % len(colors)]
        color_val_loss = colors[(idx * 4 + 1) % len(colors)]
        color_train_acc = colors[(idx * 4 + 2) % len(colors)]
        color_val_acc = colors[(idx * 4 + 3) % len(colors)]

        color_name_train_loss = color_names[(idx * 4) % len(color_names)]
        color_name_val_loss = color_names[(idx * 4 + 1) % len(color_names)]
        color_name_train_acc = color_names[(idx * 4 + 2) % len(color_names)]
        color_name_val_acc = color_names[(idx * 4 + 3) % len(color_names)]

        line_train_loss, = axes[0].plot(model.train_loss_history,
                                        color=color_train_loss,
                                        linewidth=2.5,
                                        alpha=0.9,
                                        linestyle='-')
        line_val_loss, = axes[0].plot(model.val_loss_history,
                                      color=color_val_loss,
                                      linewidth=2.5,
                                      alpha=0.9,
                                      linestyle='-')

        line_train_acc, = axes[1].plot(model.train_acc_history,
                                       color=color_train_acc,
                                       linewidth=2.5,
                                       alpha=0.9,
                                       linestyle='-')
        line_val_acc, = axes[1].plot(model.val_acc_history,
                                     color=color_val_acc,
                                     linewidth=2.5,
                                     alpha=0.9,
                                     linestyle='-')

        legend_handles_upper.extend([line_train_loss, line_val_loss, line_train_acc, line_val_acc])
        legend_labels_upper.extend([
            f'{name} train loss ({color_name_train_loss})',
            f'{name} val loss ({color_name_val_loss})',
            f'{name} train acc ({color_name_train_acc})',
            f'{name} val acc ({color_name_val_acc})'
        ])

    axes[0].set_title('Loss во время обучения', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Эпоха', fontsize=10)
    axes[0].set_ylabel('Loss (MSE)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=9)

    axes[1].set_title('Accuracy во время обучения', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Эпоха', fontsize=10)
    axes[1].set_ylabel('Accuracy', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=9)

    axes[2].axis('off')
    axes[2].set_title('Легенда цветов (верхние графики)', fontsize=11, fontweight='bold', pad=20)

    legend = axes[2].legend(legend_handles_upper, legend_labels_upper,
                            loc='center', fontsize=9, framealpha=0.9,
                            handlelength=2, handletextpad=1)
    legend.get_frame().set_facecolor('whitesmoke')
    legend.get_frame().set_edgecolor('gray')

    bar_width = 0.35
    x_positions = np.arange(n_models) * 2

    from matplotlib.patches import Patch

    for idx, (name, model) in enumerate(models_dict.items()):
        color_train_loss = colors[(idx * 4) % len(colors)]
        color_val_loss = colors[(idx * 4 + 1) % len(colors)]
        color_train_acc = colors[(idx * 4 + 2) % len(colors)]
        color_val_acc = colors[(idx * 4 + 3) % len(colors)]

        color_name_train_loss = color_names[(idx * 4) % len(color_names)]
        color_name_val_loss = color_names[(idx * 4 + 1) % len(color_names)]
        color_name_train_acc = color_names[(idx * 4 + 2) % len(color_names)]
        color_name_val_acc = color_names[(idx * 4 + 3) % len(color_names)]

        final_train_loss = model.train_loss_history[-1]
        final_val_loss = model.val_loss_history[-1]
        final_train_acc = model.train_acc_history[-1]
        final_val_acc = model.val_acc_history[-1]

        axes[3].bar(x_positions[idx] - bar_width / 2, final_train_loss,
                    width=bar_width,
                    color=color_train_loss,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1)
        axes[3].bar(x_positions[idx] + bar_width / 2, final_val_loss,
                    width=bar_width,
                    color=color_val_loss,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1)

        axes[4].bar(x_positions[idx] - bar_width / 2, final_train_acc,
                    width=bar_width,
                    color=color_train_acc,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1)
        axes[4].bar(x_positions[idx] + bar_width / 2, final_val_acc,
                    width=bar_width,
                    color=color_val_acc,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1)

        patch_train_loss = Patch(color=color_train_loss, alpha=0.8,
                                 label=f'{name} train loss ({color_name_train_loss})')
        patch_val_loss = Patch(color=color_val_loss, alpha=0.8,
                               label=f'{name} val loss ({color_name_val_loss})')
        patch_train_acc = Patch(color=color_train_acc, alpha=0.8,
                                label=f'{name} train acc ({color_name_train_acc})')
        patch_val_acc = Patch(color=color_val_acc, alpha=0.8,
                              label=f'{name} val acc ({color_name_val_acc})')

        legend_handles_lower.extend([patch_train_loss, patch_val_loss, patch_train_acc, patch_val_acc])

    axes[3].set_title('Финальные значения Loss', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Модель', fontsize=10)
    axes[3].set_ylabel('Loss', fontsize=10)
    axes[3].set_xticks(x_positions)
    axes[3].set_xticklabels(model_names, fontsize=10, fontweight='bold')
    axes[3].grid(True, alpha=0.3, axis='y')
    axes[3].tick_params(labelsize=9)

    axes[4].set_title('Финальные значения Accuracy', fontsize=12, fontweight='bold')
    axes[4].set_xlabel('Модель', fontsize=10)
    axes[4].set_ylabel('Accuracy', fontsize=10)
    axes[4].set_xticks(x_positions)
    axes[4].set_xticklabels(model_names, fontsize=10, fontweight='bold')
    axes[4].grid(True, alpha=0.3, axis='y')
    axes[4].tick_params(labelsize=9)

    for idx, x_pos in enumerate(x_positions):
        axes[3].text(x_pos - bar_width / 2, -0.02, 'Train',
                     ha='center', va='top', fontsize=8, fontweight='bold',
                     transform=axes[3].transData)
        axes[3].text(x_pos + bar_width / 2, -0.02, 'Val',
                     ha='center', va='top', fontsize=8, fontweight='bold',
                     transform=axes[3].transData)

        axes[4].text(x_pos - bar_width / 2, -0.02, 'Train',
                     ha='center', va='top', fontsize=8, fontweight='bold',
                     transform=axes[4].transData)
        axes[4].text(x_pos + bar_width / 2, -0.02, 'Val',
                     ha='center', va='top', fontsize=8, fontweight='bold',
                     transform=axes[4].transData)

    for idx, (name, model) in enumerate(models_dict.items()):
        final_train_loss = model.train_loss_history[-1]
        final_val_loss = model.val_loss_history[-1]
        final_train_acc = model.train_acc_history[-1]
        final_val_acc = model.val_acc_history[-1]

        axes[3].text(x_positions[idx] - bar_width / 2, final_train_loss + 0.005,
                     f'{final_train_loss:.4f}',
                     ha='center', va='bottom',
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        axes[3].text(x_positions[idx] + bar_width / 2, final_val_loss + 0.005,
                     f'{final_val_loss:.4f}',
                     ha='center', va='bottom',
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        axes[4].text(x_positions[idx] - bar_width / 2, final_train_acc + 0.005,
                     f'{final_train_acc:.4f}',
                     ha='center', va='bottom',
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        axes[4].text(x_positions[idx] + bar_width / 2, final_val_acc + 0.005,
                     f'{final_val_acc:.4f}',
                     ha='center', va='bottom',
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    axes[5].axis('off')
    axes[5].set_title('Легенда цветов (нижние графики)', fontsize=11, fontweight='bold', pad=20)

    legend_lower = axes[5].legend(handles=legend_handles_lower,
                                  loc='center', fontsize=9, framealpha=0.9,
                                  handlelength=2, handletextpad=1)
    legend_lower.get_frame().set_facecolor('whitesmoke')
    legend_lower.get_frame().set_edgecolor('gray')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(models_dict, X_test, y_test, model_names=None):
    if model_names is None:
        model_names = list(models_dict.keys())

    n_models = len(models_dict)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    fig.suptitle('Матрицы ошибок (Confusion Matrices)', fontsize=16, fontweight='bold')

    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict_class(X_test)
        y_true = y_test

        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    ax=axes[idx], cbar=False,
                    xticklabels=['Not Dropout', 'Dropout'],
                    yticklabels=['Not Dropout', 'Dropout'])

        axes[idx].set_title(f'{name} Model\nAccuracy: {np.mean(y_pred == y_true):.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.tight_layout()
    plt.show()


def plot_roc_curves(models_dict, X_test, y_test, model_names=None):
    if model_names is None:
        model_names = list(models_dict.keys())

    plt.figure(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models_dict)))

    for idx, (name, model) in enumerate(models_dict.items()):
        y_proba = model.predict_proba(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[idx], lw=2,
                 label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_predictions_comparison(models_dict, X_sample, y_sample, model_names=None):
    if model_names is None:
        model_names = list(models_dict.keys())

    n_models = len(models_dict)
    n_samples = min(8, len(X_sample))

    fig, axes = plt.subplots(n_models, 1, figsize=(16, 5 * n_models))
    if n_models == 1:
        axes = [axes]

    fig.suptitle('Сравнение предсказаний: реальные значения vs предсказания моделей',
                 fontsize=15, fontweight='bold')

    sample_indices = np.random.choice(len(X_sample), n_samples, replace=False)

    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred_proba = model.predict_proba(X_sample[sample_indices])
        y_pred_class = model.predict_class(X_sample[sample_indices])
        y_true = y_sample[sample_indices]

        x_pos = np.arange(n_samples)

        width = 0.25
        offset = width * 1.2

        bars_real = axes[idx].bar(x_pos - offset / 2, y_true, width,
                                  label='Реальные значения',
                                  color='#1f77b4',
                                  alpha=0.9,
                                  edgecolor='navy',
                                  linewidth=2)

        bars_pred = axes[idx].bar(x_pos + offset / 2, y_pred_proba, width,
                                  label='Предсказания',
                                  color='#ff7f0e',
                                  alpha=0.9,
                                  edgecolor='darkorange',
                                  linewidth=2)

        for j, (bar_real, bar_pred) in enumerate(zip(bars_real, bars_pred)):
            axes[idx].text(bar_real.get_x() + bar_real.get_width() / 2,
                           bar_real.get_height() + 0.02,
                           f'{y_true[j]:.2f}',
                           ha='center', va='bottom',
                           fontsize=9, fontweight='bold',
                           color='blue')

            axes[idx].text(bar_pred.get_x() + bar_pred.get_width() / 2,
                           bar_pred.get_height() + 0.02,
                           f'{y_pred_proba[j]:.2f}',
                           ha='center', va='bottom',
                           fontsize=9, fontweight='bold',
                           color='darkorange')

            if (y_true[j] > 0.5 and y_pred_proba[j] < 0.5) or \
                    (y_true[j] < 0.5 and y_pred_proba[j] > 0.5):
                rect_x = bar_real.get_x() - 0.05
                rect_y = -0.05
                rect_width = bar_pred.get_x() + bar_pred.get_width() - bar_real.get_x() + 0.1
                rect_height = max(bar_real.get_height(), bar_pred.get_height()) + 0.15

                axes[idx].add_patch(plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                                  fill=False, edgecolor='red', linewidth=2,
                                                  linestyle='--', alpha=0.7, zorder=3))

                axes[idx].text(bar_real.get_x() + rect_width / 2, rect_y + rect_height - 0.02,
                               'Ошибка классификации!',
                               ha='center', va='top',
                               fontsize=8, fontweight='bold', color='red',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        axes[idx].set_ylabel('Вероятность отчисления', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'Модель: {name} | MSE: {np.mean((y_pred_proba - y_true) ** 2):.4f}',
                            fontsize=12, fontweight='bold', pad=15)
        axes[idx].set_xticks(x_pos)

        labels = []
        for i, idx_sample in enumerate(sample_indices):
            true_class = "Отчисляется" if y_true[i] > 0.5 else "Не отчисляется"
            pred_class = "Отчисляется" if y_pred_proba[i] > 0.5 else "Не отчисляется"
            status = "✓" if true_class == pred_class else "✗"
            labels.append(f'Студент {idx_sample}\n{true_class} → {pred_class}\n{status}')

        axes[idx].set_xticklabels(labels, fontsize=9, rotation=0, ha='center')
        axes[idx].legend(loc='upper right', fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[idx].set_ylim([0, 1.2])

        axes[idx].axhline(y=0.5, color='green', linestyle='-', alpha=0.3, linewidth=2)
        axes[idx].axhline(y=0.3, color='orange', linestyle=':', alpha=0.2, linewidth=1)
        axes[idx].axhline(y=0.7, color='orange', linestyle=':', alpha=0.2, linewidth=1)

        axes[idx].axhspan(0.7, 1.0, alpha=0.1, color='red', label='Высокий риск')
        axes[idx].axhspan(0.3, 0.7, alpha=0.1, color='yellow', label='Средний риск')
        axes[idx].axhspan(0.0, 0.3, alpha=0.1, color='green', label='Низкий риск')

        axes[idx].text(-0.8, 0.52, 'Порог 0.5', fontsize=8, color='green',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()


def plot_error_distribution(models_dict, X_test, y_test, model_names=None):
    if model_names is None:
        model_names = list(models_dict.keys())

    n_models = len(models_dict)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    fig.suptitle('Распределение ошибок предсказаний', fontsize=14, fontweight='bold')

    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict_proba(X_test)
        errors = y_pred - y_test

        axes[idx].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
        axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        axes[idx].set_title(f'{name} Model\nMean Error: {np.mean(errors):.4f}')
        axes[idx].set_xlabel('Prediction Error')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_model_comparison(simple_model, complex_model, X_test, y_test):
    print("=" * 80)
    print("ВИЗУАЛИЗАЦИЯ МЕТРИК НЕЙРОННЫХ СЕТЕЙ")
    print("=" * 80)

    models = {
        'Simple NN': simple_model,
        'Complex NN': complex_model
    }

    print("\n1. Графики истории обучения...")
    plot_training_history(models)

    print("\n2. Матрицы ошибок (Confusion Matrices)...")
    plot_confusion_matrices(models, X_test, y_test)

    print("\n3. ROC-кривые...")
    plot_roc_curves(models, X_test, y_test)

    print("\n4. Сравнение предсказаний на примерах...")
    plot_predictions_comparison(models, X_test, y_test)

    print("\n5. Распределение ошибок...")
    plot_error_distribution(models, X_test, y_test)

    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА МЕТРИК")
    print("=" * 80)

    metrics_summary = []

    for name, model in models.items():
        y_pred = model.predict_proba(X_test)
        y_pred_class = model.predict_class(X_test)

        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        accuracy = np.mean(y_pred_class == y_test)

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        metrics_summary.append({
            'Model': name,
            'MSE': mse,
            'MAE': mae,
            'Accuracy': accuracy,
            'AUC': roc_auc,
            'Final Train Loss': model.train_loss_history[-1] if hasattr(model, 'train_loss_history') else 'N/A',
            'Final Val Loss': model.val_loss_history[-1] if hasattr(model, 'val_loss_history') else 'N/A'
        })

    summary_df = pd.DataFrame(metrics_summary)
    print(
        "\n" + summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))

    return summary_df


if __name__ == "__main__":
    print("Загрузка данных...")
    data = pd.read_csv("student_dropout_dataset.csv")
    X = data[["gpa", "attendance", "grade", "behavior"]].values
    y = data["dropout_probability"].values

    y_binary = (y > 0.5).astype(int)

    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X_normalized[:split_idx], X_normalized[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    y_train_binary, y_test_binary = y_binary[:split_idx], y_binary[split_idx:]

    print(f"Dataset size: {len(X)} samples")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ КОМПЛЕКСНОЙ НЕЙРОННОЙ СЕТИ")
    print("=" * 80)
    complex_model = NeuralNetwork(layers=[4, 32, 16, 8, 1], learning_rate=0.1)
    complex_model.train(X_train, y_train_binary, epochs=500, batch_size=64,
                        X_val=X_test, y_val=y_test_binary)

    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ ПРОСТОЙ НЕЙРОННОЙ СЕТИ")
    print("=" * 80)
    simple_model = SimpleNeuralNetwork(input_size=4, hidden_size=16, output_size=1)
    simple_model.train(X_train, y_train_binary, epochs=100, batch_size=32, learning_rate=0.1,
                       X_val=X_test, y_val=y_test_binary)

    complex_X_mean = X.mean(axis=0)
    complex_X_std = X.std(axis=0)

    summary_df = visualize_model_comparison(simple_model, complex_model, X_test, y_test_binary)

    print("\n" + "=" * 80)
    print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ ДЛЯ НОВЫХ СТУДЕНТОВ")
    print("=" * 80)

    new_students = np.array([
        [45, 60, 7, 4],
        [85, 95, 9, 9],
        [65, 80, 8, 6],
        [30, 40, 3, 2],
        [95, 98, 10, 10]
    ])

    print("\n" + "-" * 80)
    print(f"{'Student':^15} {'Simple NN':^15} {'Complex NN':^15} {'Risk Level':^15}")
    print("-" * 80)

    for i, student in enumerate(new_students):
        student_normalized = (student - complex_X_mean) / complex_X_std

        simple_prob = simple_model.predict_proba(student_normalized.reshape(1, -1))[0]
        complex_prob = complex_model.predict_proba(student_normalized.reshape(1, -1))[0]


        def get_risk_level(prob):
            if prob < 0.3:
                return "Low"
            elif prob < 0.7:
                return "Medium"
            else:
                return "High"


        simple_risk = get_risk_level(simple_prob)
        complex_risk = get_risk_level(complex_prob)

        print(f"{f'Student {i + 1}':^15} {simple_prob:^15.3f} {complex_prob:^15.3f} "
              f"{f'{simple_risk}/{complex_risk}':^15}")

    print("-" * 80)
    print("\nСохранение моделей...")

    complex_model_data = {
        'weights': complex_model.weights,
        'biases': complex_model.biases,
        'layers': complex_model.layers,
        'learning_rate': complex_model.learning_rate,
        'train_loss_history': complex_model.train_loss_history,
        'val_loss_history': complex_model.val_loss_history
    }

    with open('complex_model.pkl', 'wb') as f:
        pickle.dump(complex_model_data, f)

    simple_model_data = {
        'w1': simple_model.w1,
        'b1': simple_model.b1,
        'w2': simple_model.w2,
        'b2': simple_model.b2,
        'input_size': simple_model.input_size,
        'hidden_size': simple_model.hidden_size,
        'output_size': simple_model.output_size,
        'train_loss_history': simple_model.train_loss_history,
        'val_loss_history': simple_model.val_loss_history
    }

    with open('simple_model_retrained.pkl', 'wb') as f:
        pickle.dump(simple_model_data, f)

    norm_params = {
        'X_mean': complex_X_mean,
        'X_std': complex_X_std
    }

    with open('normalization_params_complex.pkl', 'wb') as f:
        pickle.dump(norm_params, f)