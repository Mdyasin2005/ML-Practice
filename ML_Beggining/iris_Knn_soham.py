import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("Iris.csv")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
'''
iris= df.columns[:-1]  
table=df.head()
print(iris)
print(table)
'''

df['Target'] = df['Species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2,
})

'''table = df.head()
print(table)'''

X= df[[ 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ]]
Y = df['Target']
X_train,  X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

'''a=(X_test)
print(a)
'''

from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors=5, p=2)

Knn.fit(X_train,Y_train)
prediction=Knn.predict(X_test)
print(f'Accuracy Score: {accuracy_score(Y_test,prediction)*100:.2f}%')
print(f'Score: {Knn.score(X_test,Y_test)*100:.2f}%')


