import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SignClassifier:
    def __init__(self, model_path="../data/model.pkl"):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def train_model(self, data_path="../data/dataset.csv"):
        if not os.path.exists(data_path):
            print("Dataset not found. Please run data_collector.py first.")
            return

        print("Loading dataset...")
        df = pd.read_csv(data_path)
        
        X = df.drop('label', axis=1)
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training KNN model...")
        # K=3 or 5 is usually good for this
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            print("Model not found. Please train first.")

    def predict(self, features):
        if self.model is None:
            return None, 0.0
        
        # Reshape features to 2D array
        features = np.array(features).reshape(1, -1)
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)
        confidence = np.max(probabilities)
        
        return prediction, confidence

if __name__ == "__main__":
    classifier = SignClassifier()
    classifier.train_model()
