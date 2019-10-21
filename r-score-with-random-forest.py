# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 21:10:21 2019

@author: tolga
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("random-forest-regression-dataset.csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%  Random Forest REg.  Sec. Predict Sec

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 200, random_state=42)
rf.fit(x,y)

y_head = rf.predict(x)

#%%  Visu.. Sec.

from sklearn.metrics import r2_score

print("Accuarry Score : ", r2_score(y,y_head))




#%%  LİNEAR REGRESSION R SCORE

import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv("linear-regression-dataset.csv", sep = ";")

plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maaş")
plt.show()


from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)


y_head = linear_reg.predict(x)

plt.plot(x,y_head,color="black")



# R Score
from sklearn.metrics import r2_score

print("Accuarry Score : ", r2_score(y,y_head))
