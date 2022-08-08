import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA
import pickle as pk

mnist = pd.read_csv('mnist_train.csv')
mnist

mnist['label'].value_counts().T

x = mnist.drop(['label'], axis=1)
y = mnist['label']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

sc=StandardScaler()

sc.fit(trainX)

trainX_scaled = sc.transform(trainX)
testX_scaled = sc.transform(testX)

logReg = LogisticRegression()

import time

start = time.time()

print(logReg.fit(trainX_scaled, trainY))

end = time.time()
print()
print('Calculation time: ' + str(end - start) + ' seconds')

y_pred = logReg.predict(testX_scaled)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

pca = PCA(.95)

pca.fit(trainX_scaled)

pca.n_components_

trainX_pca = pca.transform(trainX_scaled)
testX_pca = pca.transform(testX_scaled)

logReg = LogisticRegression()

import time

start = time.time()

print(logReg.fit(trainX_pca, trainY))

end = time.time()
print()
print('Calculation time: ' + str(end - start) + ' seconds')

y_pred = logReg.predict(testX_pca)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

pca = PCA(.80)

pca.fit(trainX_scaled)

pca.n_components_

trainX_pca = pca.transform(trainX_scaled)
testX_pca = pca.transform(testX_scaled)

logReg = LogisticRegression()

import time

start = time.time()

print(logReg.fit(trainX_pca, trainY))

end = time.time()
print()
print('Calculation time: ' + str(end - start) + ' seconds')

y_pred = logReg.predict(testX_pca)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))