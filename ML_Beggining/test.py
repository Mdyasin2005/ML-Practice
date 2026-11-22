import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("advertising.csv")
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#print(y_pred)
mae=mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
residual=y_test - y_pred
plt.figure(figsize=(7,5))
plt.scatter(y_pred, residual)
plt.axhline(y=0)
plt.title("Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.show()
print(model.predict([[100, 20, 10]]))