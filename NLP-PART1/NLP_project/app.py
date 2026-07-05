import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
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

# Download NLTK files
nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    page_icon="🍽",
    layout="wide"
)

st.title("🍽 Restaurant Review Sentiment Analysis")
st.write("Compare Multiple Machine Learning Classification Models")

uploaded_file = st.file_uploader(
    "Upload Restaurant_Reviews.tsv",
    type=["tsv"]
)

if uploaded_file is not None:

    dataset = pd.read_csv(
        uploaded_file,
        delimiter="\t",
        quoting=3
    )

    st.subheader("Dataset Preview")
    st.dataframe(dataset.head())

    if st.button("Train Models"):

        with st.spinner("Preprocessing Text..."):

            corpus = []

            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))

            for review in dataset['Review']:

                review = re.sub('[^a-zA-Z]', ' ', review)
                review = review.lower()
                review = review.split()

                review = [
                    lemmatizer.lemmatize(word)
                    for word in review
                    if word not in stop_words
                ]

                review = ' '.join(review)

                corpus.append(review)

            tfidf = TfidfVectorizer()

            X = tfidf.fit_transform(corpus).toarray()

            y = dataset.iloc[:,1].values

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=0
            )

        st.success("Preprocessing Completed")

        models = {

            'Logistic Regression':
                LogisticRegression(max_iter=1000),

            'Naive Bayes':
                MultinomialNB(),

            'Bernoulli NB':
                BernoulliNB(),

            'Decision Tree':
                DecisionTreeClassifier(random_state=0),

            'Random Forest':
                RandomForestClassifier(random_state=0),

            'Extra Trees':
                ExtraTreesClassifier(random_state=0),

            'Gradient Boosting':
                GradientBoostingClassifier(random_state=0),

            'Support Vector Machine':
                SVC(),

            'KNN':
                KNeighborsClassifier(),

            'SGD Classifier':
                SGDClassifier(random_state=0),

            'ANN (MLP)':
                MLPClassifier(
                    hidden_layer_sizes=(100,),
                    max_iter=500
                ),

            'XGBoost':
                xgb.XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss'
                ),

            'LightGBM':
                lgb.LGBMClassifier()
        }

        results = []

        progress = st.progress(0)

        total = len(models)

        for i, (name, model) in enumerate(models.items()):

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            results.append({

                "Model": name,

                "Accuracy": accuracy_score(y_test, y_pred),

                "Precision": precision_score(y_test, y_pred),

                "Recall": recall_score(y_test, y_pred),

                "F1 Score": f1_score(y_test, y_pred)

            })

            progress.progress((i + 1) / total)

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="Accuracy",
            ascending=False
        )

        st.success("Training Completed")

        st.subheader("Model Comparison")

        st.dataframe(results_df)

        best_model = results_df.iloc[0]

        st.subheader("🏆 Best Model")

        st.success(
            f"{best_model['Model']} "
            f"with Accuracy = {best_model['Accuracy']:.4f}"
        )

        st.subheader("Accuracy Comparison")

        fig, ax = plt.subplots(figsize=(10,6))

        ax.bar(
            results_df["Model"],
            results_df["Accuracy"]
        )

        plt.xticks(rotation=90)

        plt.ylabel("Accuracy")

        plt.tight_layout()

        st.pyplot(fig)

        csv = results_df.to_csv(index=False)

        st.download_button(
            "Download Results",
            csv,
            file_name="classification_results.csv",
            mime="text/csv"
        )