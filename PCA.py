import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import pandas as pd
import sys


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

    data = np.mat(raw_data)

    return data, labels

def PCA(data,num_components):
    means_mat = np.mean(data,axis=0)
    data = data - means_mat
    covMat = np.cov(data.T)
    eigenVals, eigenVectors = eig(covMat)

    #Get the top 'x' eigenVectors
    idx = np.argsort(eigenVals)
    idx = idx[:-(num_components+1):-1]
    eigenVectorsSorted = eigenVectors[:,idx]
    print(eigenVectorsSorted)
    finalData = data * eigenVectorsSorted

    return finalData

def plotGraph(filename, finalData,labels):
    #create dataframe and group based on labels
    df = pd.DataFrame(dict(x=np.asarray(finalData.T[0])[0], y=np.asarray(finalData.T[1])[0], label=labels))
    groups = df.groupby('label')

    #create the subplots
    fig, ax = plt.subplots()
    ax.margins(0.05)

    #plot all datapoints
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name)

    ax.legend()
    ax.set_title('Algorithm: PCA\n Input file: ' + filename)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')

    plt.savefig('PCA_' + filename + ".png", dpi=300)
    plt.show()

def main():

    try:
        filename = sys.argv[1]
    except:
        print("Usage: python PCA.py <inputfile> \n Example: python PCA.py pca_a.txt")
        print("No filename given, exiting!")
        exit(-1)

    dataset, labels = loadDataset(filename)
    # numRows, numCols = np.shape(dataset)
    finalData = PCA(dataset,2)
    plotGraph(filename,finalData,labels)

if __name__ == '__main__':
    main()