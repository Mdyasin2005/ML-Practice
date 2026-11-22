import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Salary_dataset.csv")
df=df.iloc[1:,1:].reset_index(drop=True)
x=df[['YearsExperience']]
y=df['Salary']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
M=LinearRegression()
M.fit(x_train, y_train)
y_pred = M.predict(x_test)
mean_abs_error=mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mean_abs_error:.2f}")
mean_sq_error=(mean_squared_error(y_test, y_pred))**0.5
print(f"Mean Squared Error: {mean_sq_error:.2f}")

plt.figure(figsize=(7,5))
plt.scatter(x, y)

plt.title("Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.show()
val=pd.DataFrame( [[2.3]],columns=['YearsExperience'])
print(M.predict(val))
