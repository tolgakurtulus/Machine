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
x1 = np.random.normal(25,5,1000)
y1 = np.random.normal(25,5,1000)

x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)

x = np.concatenate((x1,x2,x3),axis = 0)
y = np.concatenate((y1,y2,y3),axis = 0)

dictionary = {"x":x,"y":y}
data = pd.DataFrame(dictionary)

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()


#%% K-Means Seç....

from sklearn.cluster import KMeans
wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,15),wcss)
plt.xlabel("Number Of (Cluster) Value")    
plt.ylabel("Wcss")
plt.show()  


#%% Level 3 Model Sec..

kmeans2 = KMeans(n_clusters=3)
clusters = kmeans2.fit_predict(data)

data["label"] = clusters

plt.scatter(data.x[data.label == 0 ], data.y[data.label == 0 ], color = "brown" )    
plt.scatter(data.x[data.label == 1 ], data.y[data.label == 1 ], color = "black" )    
plt.scatter(data.x[data.label == 2 ], data.y[data.label == 2 ], color = "pink" )    
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color = "blue")
plt.show()












