import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("Iris.csv")
X = df.drop(["Id", "Species"], axis=1).values
y = df["Species"].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior = {}
        self.mean = {}
        self.var = {}
        for i in self.classes:
            temp = X[y == i]
            self.prior[i] = len(temp) / len(X)
            self.mean[i] = np.mean(temp, axis=0)
            self.var[i] = np.var(temp, axis=0) + 1e-9
    def predict(self, X):
        pred = []
        for i in X:
            prob = {}
            for j in self.classes:
                m = self.mean[j]
                v = self.var[j]
                temp = (1 / np.sqrt(2 * np.pi * v)) * np.exp(-((i - m) ** 2) / (2 * v))
                prob[j] = np.log(self.prior[j]) + np.sum(np.log(temp))
            pred.append(max(prob, key=prob.get))
        return np.array(pred)
model = NaiveBayes()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)