import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\hp\Downloads\logit classification.csv')

x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print(ac)

from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred)
print(cr)


bias=classifier.score(x_train,y_train)
print('bias=',bias)

variance=classifier.score(x_test,y_test)

print('var=',variance)




#Now we gonna see the visulization 

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#after execute this graph we will get the good graph & lets try to understand and analyse the graph step by step
#we have points like red point & green points, all these points are our observation of the training part 
#each of the user's age is estimated at the x-axis & estimated salary are estimated at y-axis
#red points are the training set observation which the dependent variable purchased equal to 0
# green points are the training set observation which the dependent variable puchased equal to 1
#red points users are didnt able to buy the suv & green points users are able to buy the xuv
#what we observed hear is users are young with low estimate salary actually didnt buy the xuv
#if you look at the users with older with high salary they will buy the xuv & xuv is family car so more older people are likely to buy this car
#now if you see some green points you can see in the red part , in this case even though older but due to less salary they are unable to buy car
#also some of the young people are also buy the car becuase of rich kid

#now what is the goal of classification & what classifer exactly do hear & how this classifier will do for this business case
#the main goal is classify the right users into right category that machine will do by logistic regression using s -curve
#the machine classify all dataset in 2 region, left region is to classify who not buy the car and the green is to classify who can buy the car
#logisti regression model ploted red pooint users are not going to buy the XUV and green point users are going to buy the xuv
#logistic regression will tells that each user in the dataset is proper classified based on age & salary
#main important thing is these are the 2 prediction (red & green) separated by straight line & the straight line is calle prediction boundry
#as we are talking logistice regression os linear model so we will requred only for 2 variable & sepated with straight line and logistic regressin is linear classifer
#we will see next classed how non-linear classifier will separte wont be straight line
#even though if you see the green point even though low salary they buy the xuv which is incorrect 
#if you see the green points are belongs to the red regions this happen becuae users are non-linear but we separate with linear model thats why we got this prediction 
#now we are looking visualiaton for traing set & next we going to see the visualization for test set
#


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#lets see the graph we get good graph we have plotted for the test data point 
#if you see the confustion matrix we saw 11 points are incorrectly predicted hear , you can count that 
#we build our first classification model in python '''



######         FUTURE PREDICTION         ##########################




dataset1=pd.read_csv(r'C:\Users\hp\Downloads\Future prediction1.csv')

d2=dataset1.copy()


dataset1=dataset1.iloc[:,[2,3]].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
M=sc.fit_transform(dataset1)

y_pred1=pd.DataFrame()

d2['y_pred1']=classifier.predict(M)

d2.to_csv('final.csv')

import os
os.getcwd()



#### ROC AND AUC CURVE   #######


from sklearn.metrics import roc_auc_score,roc_curve
y_pred_prob=classifier.predict_proba(x_test)[:,1]

auc_score=roc_auc_score(y_test,y_pred_prob)
auc_score

fpr,tpr,thresholds=roc_curve(y_test,y_pred_prob)


plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label=f'logistic regression(AUC={auc_score:.2f})')
plt.plot([0,1],[0,1],'r--')  #random classifier line
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()