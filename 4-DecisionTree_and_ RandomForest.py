#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')


#regresyon uygulanacak verileri ayırma
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#verileri numpy dizisine dönüştürme
X = x.values
Y = y.values



#Polinomal regresyon(2.dereceden)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)

#polinomal regresyon görselleştirmesi
plt.scatter(X,Y,color = 'black')
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()


#Polinomal regresyon (4.dereceden)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)

#polinomal regresyon görselleştirmesi
plt.scatter(X,Y,color = 'black')
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

#Polinomal regresyon ile tahmin yapılması


print(lin_reg.predict(poly_reg.fit_transform([[8]])))
print(lin_reg.predict(poly_reg.fit_transform([[11]])))

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X),color='blue')
plt.show()

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
# n_estimators = kaç tane karar ağacı(DecisionTreeRegressor) çizileceği belirlenir

rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))
plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='Blue')









