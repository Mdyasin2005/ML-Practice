# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report, confusion_matrix , accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# %%
df=pd.read_csv('Iris.csv')
X=df.drop(['Id','Species'],axis=1)
Y=df['Species']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# %%
accuracy_list=[]
for k in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    Y_pred=knn.predict(X_test)
    accuracy= accuracy_score(Y_test,Y_pred) * 100
    
    accuracy_list.append(accuracy)
    classification_rep=classification_report(Y_test,Y_pred)
    confusion_mat=confusion_matrix(Y_test,Y_pred)
    accuracy_list.append(accuracy)

    #print(f'For k={k}:')
    #print(f'Accuracy Score: {accuracy:.2f}%')
    #print('Classification Report:')
    #print(classification_rep)


# %%
model_knn=KNeighborsClassifier(n_neighbors=7)
model_knn.fit(X_train,Y_train)
y_predict=model_knn.predict(X_test)
print(f'Accuracy Score: {accuracy_score(Y_test,y_predict)*100:.2f}%')


# %%
comparison = pd.DataFrame({
    "Actual": Y_test,
    "Predicted": y_predict
})

print(comparison)

    


