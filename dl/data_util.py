import sys
import numpy as np
import string
from sklearn import decomposition
from keras.utils import np_utils
from sklearn.externals import joblib


class DataLoader:
    def __init__(self, label_dim=43, normalize=False, istrain=True, pca=True, pca_dim=100, datapath="./data/train.txt", labelpath="./data/train_label.txt"):
        self.istrain = istrain
        self.pca = pca
        self.pca_dim = pca_dim
        self.normalize = normalize
        self.label_dim = label_dim
        self.data = self.loaddata(datapath)
        self.label = self.loadlabel(labelpath)

    def loaddata(self, datapath):
        f = open(datapath,'r')
        data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].split('\t')
            data[i] = [float(d) for d in data[i]]
        data = np.array(data).astype('float32')
        f.close()
        if self.pca:
            loadpca = decomposition.PCA(n_components=self.pca_dim)
            if self.istrain:
                loadpca.fit(data)
                joblib.dump(loadpca, "./model/pca.m")
            if not self.istrain:
                loadpca = joblib.load("./model/pca.m")
            data = loadpca.transform(data)
        if self.normalize:
            data = self.norm(data)
        return data

    def loadlabel(self, labelpath):
        f = open(labelpath,'r')
        label = f.readlines()
        for i in range(len(label)):
            label[i] = label[i].split('\t')
            label[i] = [float(d) for d in label[i]]
        label = np.array(label).flatten().astype('float32')
        label = np_utils.to_categorical(label, self.label_dim)
        f.close()
        return label

    def norm(self, x):
        mean = x.mean()
        std = x.std()
        for xx in x:
            xx = (xx - mean) / std
        return x

if __name__ == '__main__':
    train = DataLoader()
    for i in train.data:
        #print i
        break
    for i in train.label:
        #print i
        break