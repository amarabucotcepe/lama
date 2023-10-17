
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import os
import joblib
from random import random, randint
import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import skops.io as sio

import xgboost as xgb

mlflow.set_experiment("teste")

with mlflow.start_run(run_name='teste5') as run:

    # Log an artifact (output file)
    if not os.path.exists("outputs/models"):
        os.makedirs("outputs/models")
    
    classes = pd.read_csv('data/classes.csv')
    produtos = pd.read_csv('data/produtos.csv')

    X_train, X_test, y_train, y_test = train_test_split(produtos['DESCRICAO'], produtos['CLASSE'],  random_state=32, shuffle=True, stratify=produtos['CLASSE'])

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    X_train_counts.shape

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_train_tf.shape

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape

    with mlflow.start_run(nested=True, run_name='naive') as run:
        mlflow.autolog()
        clf = MultinomialNB().fit(X_train_tfidf, y_train)

        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])

        text_clf.fit(X_train, y_train)

        predicted = text_clf.predict(X_test)
        acc = metrics.accuracy_score(y_test, predicted)
        log_metric("accuracy", acc)
        sio.dump(clf, 'outputs/models/MultinomialNB.skops')

    with mlflow.start_run(nested=True, run_name='sgd') as run:
        mlflow.autolog()
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                alpha=1e-3, random_state=42,
                                max_iter=5, tol=None)),
        ])

        text_clf.fit(X_train, y_train)
        predicted = text_clf.predict(X_test)
        acc = metrics.accuracy_score(y_test, predicted)
        log_metric("accuracy", acc)

        sio.dump(text_clf, 'outputs/models/SGDClassifier.skops')
        

    # print(metrics.classification_report(y_test, predicted,
    #     ))

    # fig, ax = plt.subplots(figsize=(10,10))
    # sns.heatmap(metrics.confusion_matrix(y_test, predicted), annot=True, fmt=".0f", ax=ax)

    le = LabelEncoder()

    y_train2 = le.fit_transform(y_train)

    y_test2 = le.transform(y_test)

    with mlflow.start_run(nested=True, run_name='xgb') as run:
        # mlflow.xgboost.autolog()
        text_clf2 = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', xgb.XGBClassifier()),
        ])

        text_clf2.fit(X_train, y_train2)

        predicted = text_clf2.predict(X_test)
        acc = metrics.accuracy_score(y_test2, predicted)
        log_metric("accuracy", acc)

        sio.dump(text_clf2, 'outputs/models/xgb.skops')
    
    log_artifacts("outputs")
    

# if __name__ == "__main__":
    # # Log a parameter (key-value pair)
    # log_param("config_value", randint(0, 100))

    # # Log a dictionary of parameters
    # log_params({"param1": randint(0, 100), "param2": randint(0, 100)})

    # # Log a metric; metrics can be updated throughout the run
    # log_metric("accuracy", random() / 2.0)
    # log_metric("accuracy", random() + 0.1)
    # log_metric("accuracy", random() + 0.2)

    # # Log an artifact (output file)
    # if not os.path.exists("outputs"):
    #     os.makedirs("outputs")
    # with open("outputs/test.txt", "w") as f:
    #     f.write("hello world!")
    # log_artifacts("outputs")