#Natural Language processing

#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv(r'C:\Users\hp\Downloads\Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#cleaning teh texts
import re
import nltk
#nltk.download('stopwords)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]

for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
#creating teh Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1,2)
)
X=tfidf.fit_transform(corpus).toarray()

y=dataset.iloc[:,1].values

#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

#predicting the test set results
y_pred=classifier.predict(X_test)

#Making the Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print(ac)

from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred)
print(cr)


bias=classifier.score(X_train,y_train)
print(bias)

variance=classifier.score(X_test,y_test)
print(variance)

'''
CASE STUDY --> model is underfitted  & we got less accuracy 

1> Implementation of tfidf vectorization , lets check bias, variance, ac, auc, roc 
2> Impletemation of all classification algorihtm (logistic, knn, randomforest, decission tree, svm, xgboost,lgbm,nb) with bow & tfidf 
4> You can also reduce or increase test sample 
5> xgboost & lgbm as well
6> you can also try the model with stopword 


6> then please add more recores to train the data more records 
7> ac ,bias, varian - need to equal scale ( no overfit & not underfitt)

'''



#Predicting best accuracy of all classification models



import pandas as pd

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Dictionary of Classification Models
models = {

    'Logistic Regression' : LogisticRegression(max_iter=1000),

    'Naive Bayes' : MultinomialNB(),

    'Bernoulli NB' : BernoulliNB(),

    'Decision Tree' : DecisionTreeClassifier(random_state=0),

    'Random Forest' : RandomForestClassifier(random_state=0),

    'Extra Trees' : ExtraTreesClassifier(random_state=0),

    'Gradient Boosting' : GradientBoostingClassifier(random_state=0),

    'Support Vector Machine' : SVC(),

    'KNN' : KNeighborsClassifier(),

    'SGD Classifier' : SGDClassifier(random_state=0),

    'ANN (MLP)' : MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),

    'XGBoost' : xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss'
    ),

    'LightGBM' : lgb.LGBMClassifier()
}

# Train and Evaluate

results=[]

for name,model in models.items():

    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    acc=accuracy_score(y_test,y_pred)

    precision=precision_score(y_test,y_pred)

    recall=recall_score(y_test,y_pred)

    f1=f1_score(y_test,y_pred)

    results.append({
        'Model':name,
        'Accuracy':acc,
        'Precision':precision,
        'Recall':recall,
        'F1 Score':f1
    })

# Display Results

results_df=pd.DataFrame(results)

results_df=results_df.sort_values(by='Accuracy',ascending=False)

print( results_df)
