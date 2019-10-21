# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:01:21 2019

@author: tolga
"""

import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv("linear-regression-dataset.csv", sep = ";")

plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maaş")
plt.show()

#%% sklearn linear regression

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

# we add values because we need to convert numpy
# We use reshape becaıse normally (14,) but need to = (14,1) su we use reshape
x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

#fit model
linear_reg.fit(x,y)

#%% predict   y=b0+b1*x This is our formula
import numpy as np
b0 = linear_reg.predict([[0]])

b1 = linear_reg.coef_

# 11 Yıllık Deneyim = b0(1663)+b1(1138)*11
print("11 Yıllık Deneyim = ",linear_reg.predict([[11]]))



#visulization model

array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  #because array is (16,)

plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array)

plt.plot(array,y_head, color="black")  #array deneyim  ----- y_head maas






