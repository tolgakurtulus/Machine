# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:40:18 2019

@author: tolga
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% Create Dataset Seç....

 # Avg = 25, Sigma = 5, Pics = 1000 
x1 = np.random.normal(25,5,100)
y1 = np.random.normal(25,5,100)

x2 = np.random.normal(55,5,100)
y2 = np.random.normal(60,5,100)

x3 = np.random.normal(55,5,100)
y3 = np.random.normal(15,5,100)

x = np.concatenate((x1,x2,x3),axis = 0)
y = np.concatenate((y1,y2,y3),axis = 0)

dictionary = {"x":x,"y":y}
data = pd.DataFrame(dictionary)

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()


#%% Dendrogram Seç....

from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data,method = "ward")
dendrogram(merg,leaf_rotation=90)
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()





#%% HC Sec..

from sklearn.cluster import AgglomerativeClustering
hiyerartical_cluster = AgglomerativeClustering(n_clusters=3, affinity= "euclidean", linkage="ward") 
cluster = hiyerartical_cluster.fit_predict(data)
data["label"] = cluster



plt.scatter(data.x[data.label == 0 ], data.y[data.label == 0 ], color = "brown" )    
plt.scatter(data.x[data.label == 1 ], data.y[data.label == 1 ], color = "black" )    
plt.scatter(data.x[data.label == 2 ], data.y[data.label == 2 ], color = "pink" )    











