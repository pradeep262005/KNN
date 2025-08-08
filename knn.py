# model.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

X = df[iris.feature_names]
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save model
with open('knn_model.pkl', 'wb') as f:
    pickle.dump((model, iris), f)
