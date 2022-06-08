#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")


#### Veri Önişleme ####

#LabeleEncoder:  Kategorik -> Numeric
play = veriler.iloc[:,-1:].values
print(play)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
play[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(play)

##### Tüm Satırlara Label Encoding Uygulamak İçin ######
from sklearn import preprocessing

veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

#OneHotEncoder: Kategorik -> Numeric

c = veriler.iloc[:,:1]
ohe = preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

### Analize Hazır Son Tabloyu Oluşturma ###

havadurumu = pd.DataFrame(data = c, index = range(14), columns=['overcast','rainy','sunny'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)

### Verilerin Egitim ve Test İcin Bölünmesi ###

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

### Çoklu Doğrusal Regresyon Uygulanması ###
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

### Tahmin değerlerinin yazdırılması ###
y_pred = regressor.predict(x_test)
print(y_pred)

### Backward Elimination Yöntemi ile İyileştirme ###

import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

## P-Value değeri yüksek olan sütunu veriden ayırma
sonveriler = sonveriler.iloc[:,1:]

## Yeni Tahminleme yapabilmek için P-Value değeri yüksek olan sütunu train ve test verisinden ayırma

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

## Elimination uygulanan veri ile tekrar tahminleme yapılması
import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)


