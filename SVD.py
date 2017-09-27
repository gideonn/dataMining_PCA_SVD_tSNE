from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import *
from scipy.stats.stats import pearsonr
from numpy import linalg as la

fileName = 'pca_c.txt'
#get total columns
with open(fileName, 'r') as f:
    numCols = len(f.readline().split('\t'))

# print("Number of columns in file {} is : {}".format(fileName,numCols))
# load data points
raw_data = loadtxt(fileName,delimiter='\t',skiprows=0,usecols=range(0,numCols-1))
samples,features = shape(raw_data)

print(samples, features)

# normalize and remove mean
# data = mat(raw_data[:, :4])


def svd(data, S=2):
    # calculate SVD
    U, s, V = linalg.svd(data)
    # Sig = mat(eye(S) * s[:S])
    # take out columns you don't need

    print("U: {} \n s: {} \n V: {}".format(U,s,V))

    newdata = U[:, :S]
    print(newdata)
    # print("NEW DATA IS : \n",newdata)
    # this line is used to retrieve dataset
    # ~ new = U[:,:2]*Sig*V[:2,:]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['blue', 'red', 'black']

    for i in range(samples):
        # print("Val of i: {}, newData[i,0]: {}".format(i, raw_data[i,1]))
        ax.scatter(newdata[i, 0], newdata[i, 1], marker='*', color=colors[(int(raw_data[i,-1]))%3])
    plt.xlabel('SVD1')
    plt.ylabel('SVD2')
    plt.show()

svd(raw_data,2)