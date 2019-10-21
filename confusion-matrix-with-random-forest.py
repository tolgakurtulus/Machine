# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% İnformation

data = pd.read_csv("knndata.csv")
data.info()
data.head()
data.describe()

#%% Cleacinig Sec..

data.drop(["id", "Unnamed: 32"],axis=1,inplace = True)


# %% Some İmportant Settings


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

# %% normalization
# (x - min(x))/(max(x)-min(x))


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=42)


#%% KNN ALgorit.. Sec..

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 1)  # n_neighbors = Key value
rf.fit(x_train,y_train)



#%% Score Print Func.. Sev..

print(" Score : {} ".format(rf.score(x_test,y_test))) 

y_pred = rf.predict(x_test)
y_true = y_test

#%% Conf Matrix Sec...

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

#%% Conf Matrix Visu.. Sec...

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm, annot = True, linewidths = 0.5, linecolor = "red", fmt =".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

























