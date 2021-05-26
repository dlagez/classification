# author: roczhang
# file: tutorial.py
# time: 2021/04/16
import numpy as np
rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB()
clf.fit(X, y)
print(clf.predict(X[2:3]))