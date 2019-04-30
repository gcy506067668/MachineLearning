import random
import numpy as np

cifar10_dir = 'datasets/cifar-10-batches-py'

class knn:
    def __init__(self):
        self.Xtr = None
        self.Ytr = None

    def __init__(self,x,y):
        self.Xtr = x
        self.Ytr = y

    def train(self, X, Y):
        self.Xtr = X
        self.Ytr = Y

    def predict(self, x, k=1):
        distances = np.sum((self.Xtr - x)**2, axis=1)
        print(distances.shape)
        k_labels = [self.Ytr[x] for x in np.argsort(distances)][:k]
        u, counts = np.unique(k_labels, return_counts=True)
        return u[np.argmax(counts)]

    #datasetspath 数据集path的list
    def predictLowMemery(self,datasetsPath,loadfunction,x,k=1):

        for path in range(0,datasetsPath.count()):
            try:
                del xtr
                del ytr
            except:
                pass

            xtr,ytr = loadfunction(path)
            distances = np.sum((xtr - x)**2, axis=1)
            # k_labels = [ytr[x] for x in np.argsort(distances)][:k]
            k_labels = [ytr[x] for x in np.argsort(distances)]
            print(k_labels.shape)

            # u, counts = np.unique(k_labels, return_counts=True)
            pass
        pass

def saveImage(x):
    img = np.reshape(x, (32, 32,3))  # Xtr['data']为图片二进制数据
    img = img.transpose(1, 2, 0)  # transpose函数->转换坐标轴，
     # (0,1,2)->保持不变，
     # (1,0,2)->x与y轴对调
     # (0,2,1)->y与z轴对调
     # ()->无参数时为转置矩阵
    imsave("./test.jpg", img)
    pass

import matplotlib.pyplot as plt
from util.data_utils import load_CIFAR_batch
from scipy.misc import imsave
import psutil
import os

if __name__ =='__main__':

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    x,y = load_CIFAR_batch("./datasets/cifar-10-batches-py/data_batch_1")

    knn = knn(x,y)
    index = y[knn.predict(x[10],9)]
    # predict label
    print("predict : "+classes[index])

    # real label
    print("real : "+classes[y[10]])


    # info = psutil.virtual_memory()
    # print("内存使用量："+str(psutil.Process(os.getpid()).memory_info().rss))
    # print(info.percent)








