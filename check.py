from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sys

X = [[0,3],[2,3],[5,6],[4,5],[3,4],[1,5],[6,7],[5,4]]
y=[]
with open("labels.txt",'r') as f:
    l = f.readline().split()
    for i in l:
        y.append(int(i))
print(y)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)

KNeighborsClassifier(n_neighbors=3)
print(neigh.predict([[1,4],[3,2]]))