import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize MLP model
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=200, random_state=42)

# Train the MLP model
mlp.fit(x_train, y_train)

# Make predictions
y_predict = mlp.predict(x_test)

# Evaluate accuracy
score = accuracy_score(y_test, y_predict)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# Save the trained model
with open('mlp_model.p', 'wb') as f:
    pickle.dump({'model': mlp}, f)

# --- Visualization ---

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_predict)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 2. Loss Curve (Training Progress)
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_, label="Training Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve Over Epochs")
plt.legend()
plt.grid(True)
plt.show()
