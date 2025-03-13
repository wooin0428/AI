import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Evaluate accuracy
score = accuracy_score(y_predict, y_test)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# Save the trained model
with open('model_randforest.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# --- Visualization ---

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_predict)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 2. Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(range(len(importances)), importances[indices], align="center", color="skyblue")
plt.xticks(range(len(importances)), indices, rotation=90)
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Random Forest")
plt.show()

# 3. Accuracy vs Number of Trees
n_estimators = [10, 50, 100, 200, 500]
accuracy_scores = []

for n in n_estimators:
    temp_model = RandomForestClassifier(n_estimators=n, random_state=42)
    temp_model.fit(x_train, y_train)
    y_pred_temp = temp_model.predict(x_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred_temp))

plt.figure(figsize=(8, 5))
plt.plot(n_estimators, accuracy_scores, marker="o", linestyle="-", color="green")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Number of Trees")
plt.grid(True)
plt.show()
