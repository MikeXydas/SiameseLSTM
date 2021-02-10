import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train_acc = pd.read_csv('../storage/datasets/q2b/plots/epochs500/run-train-tag-epoch_accuracy.csv')
val_acc = pd.read_csv('../storage/datasets/q2b/plots/epochs500/run-validation-tag-epoch_accuracy.csv')

train_loss = pd.read_csv('../storage/datasets/q2b/plots/epochs500/run-train-tag-epoch_loss.csv')
val_loss = pd.read_csv('../storage/datasets/q2b/plots/epochs500/run-validation-tag-epoch_loss.csv')

fig = plt.figure(figsize=(12, 5))


epochs = 500

plt.subplot(1, 2, 1)
plt.plot(np.arange(epochs), train_loss[:epochs].Value, label="Train", linewidth=3) #, marker='o')
plt.plot(np.arange(epochs), val_loss[:epochs].Value, label="Validation", linewidth=3) #, marker='o')
plt.legend(prop={'size': 18}, markerscale=5)
plt.title('Loss vs. Epochs', fontsize=22)
plt.xlabel("Epoch", fontsize=19)
plt.ylabel("BCE Loss", fontsize=19)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(np.arange(epochs), train_acc[:epochs].Value, label="Train", linewidth=3)#, marker='o')
plt.plot(np.arange(epochs), val_acc[:epochs].Value, label="Validation", linewidth=3)#, marker='o')
plt.legend(prop={'size': 18}, markerscale=5)
plt.title('Accuracy vs. Epochs', fontsize=22)
plt.xlabel("Epoch", fontsize=19)
plt.ylabel("Accuracy", fontsize=19)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()

fig.tight_layout()
# plt.show()
plt.savefig('../storage/datasets/q2b/plots/lstm_dense_features_curves_500.png', bbox_inches='tight')
