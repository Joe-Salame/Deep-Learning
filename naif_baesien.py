import numpy as np
import pandas as pd
import tensorflow as tf
import urllib
import matplotlib.pyplot as plt
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

data = pd.read_csv('spam.csv')
x = data.drop("Spam", axis=1)
y = data["Spam"]

#nous utiliserons le module train test split pour diviser l'ensemble de données en sections d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .33, random_state =17)

#nous importons le module Gaussian, pour ajuster le modèle, nous pouvons passer x_train et y_train.
GausNB = GaussianNB()
y_expect = y_test
GausNB.fit(X_train, y_train)

#Le accuracy_score suivant reflète le succès avec lequel notre modèle Gaussian a prédit a l'aide des données de test.
y_pred = GausNB.predict(X_test)
print (f"Accuracy score: {accuracy_score(y_expect, y_pred)}" )