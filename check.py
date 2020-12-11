from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sys
import time

# X = [[0,3],[2,3],[5,6],[4,5],[3,4],[1,5],[6,7],[5,4]]
# y=[]
TEST_SIZE = 20
KERNEL_SIZE = 28
DIM_SIZE = 10

# load labels
y=[]
with open("labels.txt",'r') as f:
    l = f.readline().split()
    for i in l:
        y.append(int(i))
print(len(y))

# load testsets
t=[]
with open("testsets.txt",'r') as f:
    l = f.readline().split()
    tmp = []
    for i in range(len(l)):
        if i % DIM_SIZE == 0:
            tmp = []
            tmp.append(float(l[i]))
        elif i % DIM_SIZE == (DIM_SIZE - 1):
            tmp.append(float(l[i]))
            t.append(tmp)
        else:
            tmp.append(float(l[i]))
print(len(t))
# load datasets
x=[]
with open("datasets.txt",'r') as f:
    l = f.readline().split()
    tmp = [[] for i in range(KERNEL_SIZE)]
    for i in range(len(l)):
        # dim = i/28
        data_num = i%KERNEL_SIZE
        tmp[data_num].append(float(l[i]))
        if i % (KERNEL_SIZE * DIM_SIZE) == (KERNEL_SIZE * DIM_SIZE - 1):
            for ele in tmp:
                x.append(ele)
            tmp = [[] for i in range(KERNEL_SIZE)]

print(len(x))
# load our predictions
our_prediction =[]
with open("outputlabel.txt",'r') as f:
    our_prediction = f.readline().split()
    our_prediction = [int(element) for element in our_prediction]
neigh = KNeighborsClassifier(n_neighbors=5)

print(len(our_prediction))
# print(x)
neigh.fit(x,y)

KNeighborsClassifier(n_neighbors=5)
print("sklearn output:")
print(neigh.predict(t))
print("our prediction:")
print(our_prediction)

start_time = time.time()
for i in range(1000):
    neigh.fit(x, y)
    neigh.predict(t)
print("--- %s seconds ---" % ((time.time() - start_time)/1000))