import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv(r'C:\Users\hp\Downloads\Salary_Data.csv')


print("Dataset shape:",dataset.shape)
                      
x=dataset.iloc[:,:-1]

y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
print(y_pred)

comparision=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
comparision

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('Salary vs experience(Test set)')
plt.xlabel('Years of Experienece')
plt.ylabel('Salary')
plt.show()


model_coef=regressor.coef_
print(model_coef)

model_const=regressor.intercept_
print(model_const)


y_12=model_coef*12+model_const
print(y_12)

y_25=model_coef*25+model_const
print(y_25)


#stats for ML

dataset.mean()

dataset['Salary'].mean()

dataset.median()

dataset['Salary'].median()

dataset.mode()

dataset.var()

dataset['Salary'].var()

dataset.std()

#ssr
y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

#sse
y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

#sst
mean_total=np.mean(dataset.values)
SST=np.sum((dataset.values-mean_total)**2)
print(SST)

#r2
r_square=1-(SSR/SST)
print(r_square)

bias=regressor.score(x_train,y_train)
print(bias)

#variance
variance=regressor.score(x_test,y_test)
print(variance)

#Coefficient of variation(cv)
#for calculating the cv we have to import a library first
from scipy.stats import variation
variation(dataset.values) #this will give cv of entire dataframe

variation(dataset['Salary']) #this will give us the cv of that particular column

#Correlation
dataset.corr()#this will give correlation of entire dataframe

dataset['Salary'].corr(dataset['YearsExperience'])# this will give us correlation between these two 

#Skewness
dataset.skew()

dataset['Salary'].skew()

#standard error
dataset.sem()

dataset['Salary'].sem()

#Zscore
#for calculating Z-score we have to import a library first
import scipy.stats as stats

dataset.apply(stats.zscore)#give z-score of entire dataframe

stats.zscore(dataset['Salary'])

#Degree of freedom



#Deployment in flask and html
#mlops(azure,googlecolab,heroku,kubarnate)

import pickle

#save the trained model to disk
filename='linear_regression_model.pkl'

#open a file in write-binary mode and dump the model
with open(filename,'wb') as file:
    pickle.dump(regressor,file)

print("Model has been pickled and saved as linear_regression_model.pkl")


import os
os.getcwd()