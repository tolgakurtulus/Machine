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

#%% Disc.. Sec..

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

#%% Visu... Sec..
#Legend means up-right corner write Bad and Good label
plt.scatter(M.radius_mean, M.texture_mean,color="blue",label="Bad", alpha = 0.4)
plt.scatter(B.radius_mean, B.texture_mean,color="grey",label="Good", alpha=0.6)
plt.xlabel("Radius")
plt.ylabel("Texture")
plt.legend()
plt.show()


# %% Some İmportant Settings


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

# %% normalization
# (x - min(x))/(max(x)-min(x))


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=42)


#%% KNN ALgorit.. Sec..

from sklearn.svm import SVC
svm = SVC(random_state = 1)  # n_neighbors = Key value
svm.fit(x_train,y_train)



#%% Score Print Func.. Sev..

print(" Score : {} ".format(svm.score(x_test,y_test))) 



#%% Find and Vis. The Best K Value With For Sec...


score_list = []
for i in range(1,100):
    svm2 = SVC(random_state = i)  # n_neighbors = Key value
    svm2.fit(x_train,y_train)
    score_list.append(svm2.score(x_test,y_test))
    
plt.plot(range(1,100),score_list)
plt.xlabel("Key Values")    
plt.ylabel("Accuarcy")
plt.show()


























