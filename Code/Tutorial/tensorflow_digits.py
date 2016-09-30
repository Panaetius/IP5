from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.cross_validation import train_test_split
import tensorflow.contrib.learn as skflow
from sklearn import metrics

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

n_classes = len(set(y_train))
classifier = skflow.DNNClassifier(hidden_units=[1000, 500, 500, 500], n_classes=n_classes)
classifier.fit(X_train, y_train, steps=200, batch_size=32)

y_pred = classifier.predict(X_test)

print(metrics.classification_report(y_true=y_test, y_pred=y_pred))