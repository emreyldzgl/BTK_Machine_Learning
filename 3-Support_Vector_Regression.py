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


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR

## kernel = rbf
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='black')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')


print(svr_reg.predict([[11]]))
print(svr_reg.predict([[8]]))

## kernel = linear
svr_reg2 = SVR(kernel='linear')
svr_reg2.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='black')
plt.plot(x_olcekli,svr_reg2.predict(x_olcekli),color='red')


print(svr_reg2.predict([[11]]))
print(svr_reg2.predict([[8]]))


## kernel = poly
svr_reg3 = SVR(kernel='poly')
svr_reg3.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='black')
plt.plot(x_olcekli,svr_reg3.predict(x_olcekli),color='green')


print(svr_reg3.predict([[11]]))
print(svr_reg3.predict([[8]]))








