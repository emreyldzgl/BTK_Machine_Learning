# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:24:33 2022

@author: emrey
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('musteriler.csv')

X = data.iloc[:,3:].values

## K-Means Algoritması ##

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)

## En İyi Bölütleyen Clusters Sayısını Belirleme için Döngü Oluşturma ##

sonuclar = []

for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

## Karşılaştırmaları Görselleştirme ##

plt.plot(range(1,11),sonuclar)
plt.show()

kmeans = KMeans (n_clusters = 4, init='k-means++', random_state= 123)
y_predict = kmeans.fit_predict(X)
print(y_predict)

plt.scatter(X[y_predict==0,0],X[y_predict==0,1],s=100,c='red')
plt.scatter(X[y_predict==1,0],X[y_predict==1,1],s=100,c='blue')
plt.scatter(X[y_predict==2,0],X[y_predict==2,1],s=100,c='green')
plt.scatter(X[y_predict==3,0],X[y_predict==3,1],s=100,c='yellow')
plt.title('KMeans')
plt.show()


## Hierarchical Clustering ##
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_predict = ac.fit_predict(X)
print(y_predict)

plt.scatter(X[y_predict==0,0],X[y_predict==0,1],s=100,c='red')
plt.scatter(X[y_predict==1,0],X[y_predict==1,1],s=100,c='blue')
plt.scatter(X[y_predict==2,0],X[y_predict==2,1],s=100,c='green')
plt.scatter(X[y_predict==3,0],X[y_predict==3,1],s=100,c='yellow')
plt.title('HClustering')
plt.show()

## Dendrogram ##
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()




