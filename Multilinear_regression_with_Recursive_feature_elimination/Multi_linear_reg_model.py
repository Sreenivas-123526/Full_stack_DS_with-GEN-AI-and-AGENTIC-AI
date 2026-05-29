import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r'C:\Users\hp\Desktop\FULLSTACK_DATASCIENCE(from 23,24 April)\11th - mlr\MLR\Investment.csv')


x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

x=pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)



from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

print(regressor.coef_)

print(regressor.intercept_)

x=np.append(arr=np.full((50,1),42467).astype(int), values=x,axis=1)

#till now we build multiple linear regression model
#what is next? i have to solve the organization problem
#we need to find out the best attribute to grow the business with less investment

import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,4,5]]

#OrdinaryLeastSquares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,5]]

#OrdinaryLeastSquares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1,2,3]]

#OrdinaryLeastSquares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_opt=x[:,[0,1,3]]

#OrdinaryLeastSquares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1]]

#OrdinaryLeastSquares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()








