import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Customer%20Purchase.csv')
X=df[['Age','Gender','Education','Review']]
y=df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=DecisionTreeClassifier(criterion="squared_error",
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42)
model.fit(X_train,y_train)