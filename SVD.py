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

    # load data points
    raw_data = np.loadtxt(fileName, delimiter='\t', skiprows=0, usecols=range(0, numCols - 1))

    data = np.mat(raw_data)

    return data, labels


def SVD(data,num_components):
    # center the data
    # means_mat = np.mean(data, axis=0)
    # data = data - means_mat

    # call the numpy's SVD function
    # U, s, V = linalg.svd(data)
    # newdata = U[:, :num_components]

    #Using scikit learns' TruncatedSVD method
    svd = TruncatedSVD(n_components=num_components).fit_transform(data)
    newdata = svd

    return newdata


def plotGraph(filename, finalData, labels):
    #Data frame for method: TSVD
    df = pd.DataFrame(dict(x=finalData[:,0], y=finalData[:,1], label=labels))

    #Data frame for method: NP SVD
    # df = pd.DataFrame(dict(x=np.asarray(finalData.T[0])[0], y=np.asarray(finalData.T[1])[0], label=labels))

    groups = df.groupby('label')
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name)
    ax.legend()
    ax.set_title('Algorithm: SVD\n Input file: ' + filename)

    plt.xlabel('SVD1')
    plt.ylabel('SVD2')

    plt.savefig('SVD_' + filename + ".png",  dpi=300)
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
    finalData = SVD(dataset,2)

    plotGraph(filename,finalData,labels)


if __name__ == '__main__':
    main()