import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.decomposition import TruncatedSVD

def loadDataset(fileName):
    labels = []
    try:
        with open(fileName, 'r') as f:
            data = f.readline().strip().split('\t')
            numCols = len(data)
            labels.append(data[numCols - 1])
            for line in f:
                # print(line)
                labels.append(line.strip().split('\t')[numCols-1])
    except FileNotFoundError:
        print("File not found, please check the filename in the argument.")
        exit(-1)
    except:
        print("Error opening the file")
        exit(-1)

    # print("Number of columns in file {} is : {}".format(fileName,numCols))
    # load data points
    raw_data = np.loadtxt(fileName, delimiter='\t', skiprows=0, usecols=range(0, numCols - 1))
    # samples, features = np.shape(raw_data)
    # data = np.mat(raw_data[:, :4])
    data = np.mat(raw_data)

    return data, labels


def SVD(data,num_components):
    # calculate SVD
    # U, s, V = linalg.svd(data)
    # newdata = np.dot(U[:, 0:num_components], np.dot(np.diagflat(s[0:num_components]), V[0:num_components, :]))
    # newdata = U[:, :num_components]
    # newdata = U[:, :num_components].dot(np.diag(s))
    # newdata = np.dot(U[:, :num_components], np.dot(np.diag(s[:num_components]), V[:num_components, :]))
    # print(newdata)
    # print(np.shape(U[:, :num_components]))
    # newdata=newdata[:,:2]
    # print(np.shape(newdata))

    svd = TruncatedSVD(n_components=num_components).fit_transform(data)

    # svd.fit(data)
    # newdata = svd.transform(data)
    # # # print(data)
    newdata = svd
    # print(np.shape(newdata[:,0]))
    # # print(np.shape(newdata[:,0]))
    return newdata


def plotGraph(numRows, finalData, dataset, labels):
    #TSVD
    df = pd.DataFrame(dict(x=finalData[:,0], y=finalData[:,1], label=labels))

    #NP SVD
    # df = pd.DataFrame(dict(x=np.asarray(finalData.T[0])[0], y=np.asarray(finalData.T[1])[0], label=labels))

    groups = df.groupby('label')
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name)
    ax.legend()

    plt.xlabel('SVD1')
    plt.ylabel('SVD2')

    plt.show()

def main():
    try:
        filename = sys.argv[1]
    except:
        print(sys.exc_info()[0])
        print("Usage: python PCA.py pca_a.txt")
        print("No filename given, exiting!")
        exit(-1)

    print("Running SVD for file: {}".format(filename))
    dataset, labels = loadDataset(filename)
    numRows, numCols = np.shape(dataset)
    finalData = SVD(dataset,2)

    # print("FINAL DATA : \n", finalData)

    plotGraph(numRows,finalData,dataset,labels)


if __name__ == '__main__':
    main()