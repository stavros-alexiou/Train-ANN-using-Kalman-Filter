from __future__ import division
from curses import nl
from sre_constants import ANY
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from knn import KNN, load_knn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import kalmann

df = pd.read_csv('breast-cancer-diagnostic-data.csv', encoding='utf-8')
print('\n\n')
print(df)

y = df['diagnosis'].values
print('\n\n')
print(y)

le = LabelEncoder()
Y = le.fit_transform(y)
Y = Y.reshape(-1, 1)
print('\n\n')
print(Y)

X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
print('\n\n')
print(X)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
print('\n\n')
print(X)

# create a KNN instance.
knn_ekf = kalmann.KNN(nu=30, ny=1, nl=32, neuron='logistic')

# train the model.
nepochs_ekf = 10
stdev = 0.2
knn_ekf.train(nepochs=nepochs_ekf, U=X, Y=Y, method='ekf', P=0.2, Q=0, R=stdev**2, pulse_T=2)

# use KNN instances as classifiers.
F_ekf = knn_ekf.classify(X, high=1, low=0)
print('\n\n')
print("Extended Kalman Filter Classification Accuracy: {}%".format(int(100*np.sum(F_ekf==Y)/len(Y))))

# Evaluation
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
ax.set_title("True Classifications", fontsize=22)
ax.scatter(X[:, 0], X[:, 1], c=Y[:,0])
plt.axis('equal')
ax = fig.add_subplot(1, 3, 2)
ax.set_title("EKF: {} epochs".format(nepochs_ekf), fontsize=22)
ax.scatter(X[:, 0], X[:, 1], c=F_ekf[:,0])
plt.axis('equal')
# ax = fig.add_subplot(1, 3, 3)
# ax.set_title("SGD: {} epochs".format(nepochs_sgd), fontsize=22)
# ax.scatter(U[:, 0], U[:, 1], c=F_sgd[:,0])
# plt.axis('equal')
plt.show()
