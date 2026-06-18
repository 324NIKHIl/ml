import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report,
confusion_matrix

iris=load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_st
ate=42)
model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: ",accuracy)

print("Iris Classification
report:\n",classification_report(y_test,y_pred,target_names=iris.target_na
mes))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
