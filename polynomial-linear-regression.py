# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:58:03 2019

@author: tolga
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("polynomial-regression.csv", sep = ";")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

#%% visu..  sec.

plt.scatter(x,y)
plt.ylabel("araba max hızı")
plt.xlabel("araba fiyat")
plt.show()


#%% linear model
from sklearn.linear_model import LinearRegression

Lr = LinearRegression()

Lr.fit(x,y)

y_head = Lr.predict(x)
plt.plot(x,y_head, color="red", label="Linear")
plt.show()

#%%  polynomial linear regression sec.

from sklearn.preprocessing import PolynomialFeatures
poly_reg =  PolynomialFeatures(degree = 4)

#%%  fit
x_poly = poly_reg.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)

#%%  visu2..  sec

y_head2 = lin_reg.predict(x_poly)


plt.plot(x,y_head2, color="black", label="Poly" )
plt.legend()
plt.show()



