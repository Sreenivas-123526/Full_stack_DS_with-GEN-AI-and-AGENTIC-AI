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