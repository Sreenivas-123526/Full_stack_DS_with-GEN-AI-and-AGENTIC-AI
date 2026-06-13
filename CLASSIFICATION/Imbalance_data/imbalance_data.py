import numpy as np
import pandas as pd
import matplotlib.pyplot as plt\

dataset=pd.read_csv(r'C:\Users\hp\Desktop\FULLSTACK_DATASCIENCE(from 23,24 April)\28th - SVM\Social_Network_Ads.csv')

x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

##to balance the imbalance data

#1.Random Under Sampling

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler()

X_train_bal, y_train_bal = rus.fit_resample(
    x_train,
    y_train
)

#2.Random Over Sampling

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()

X_train_bal, y_train_bal = ros.fit_resample(
    x_train,
    y_train
)

#3.SMOTE (Most commonly used)(Synthetic Minority Over-sampling Technique)

from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_train_bal, y_train_bal = smote.fit_resample(
    x_train,
    y_train
)

#4. Class Weight (No data modification)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight='balanced'
)

model.fit(x_train, y_train)