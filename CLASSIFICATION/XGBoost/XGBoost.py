#XGBoost

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv(r'C:\Users\hp\Downloads\Churn_Modelling.csv')

X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

#Label encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
X[:,2]=encoder.fit_transform(X[:,2])

#One hot encoding the "Geography" column

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],
                                    remainder='passthrough')

X=np.array(ct.fit_transform(X))


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from xgboost import XGBClassifier
classsifier=XGBClassifier()
classsifier.fit(X_train,y_train)

y_pred=classsifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print(ac)

from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred)
print(cr)

bias=classsifier.score(X_train,y_train)
print(bias)

variance=classsifier.score(X_test,y_test)
print(variance)