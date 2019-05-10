from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

data_path = "./data/sale_data.csv"

def loaddata(data_path):
    return pd.read_csv(data_path)


def preprocess(data):
    data_mean = data.mean()
    data_std = data.std()
    data_zs = 1.0*(data - data_mean)/data_std
    return data_zs
    pass

def recoverproprocess(data):

    pass

def dbscan(data,eps,min_sam):

    data = data[["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]]
    DBmodel = DBSCAN(eps=eps,  # 邻域半径0.5
           min_samples=min_sam,    # 最小样本点数，MinPts 5
           metric='euclidean',
           metric_params=None,
           algorithm='auto', # 'auto','ball_tree','kd_tree','brute',4个可选的参数 寻找最近邻点的算法，例如直接密度可达的点
           leaf_size=30, # balltree,cdtree的参数
           p=None,
           n_jobs=1)
    y_pre = DBmodel.fit_predict(data)
    return DBmodel,y_pre
    pass

if __name__ =='__main__':
    data = loaddata(data_path)
    print(data)
    data = preprocess(data)
    print(data)
    colors = ['1','red', 'blue', 'green', 'orange', 'black']


    # for i in range(2,7): #min_sam
    #     line = []
    #     for j in range(1,6):       #eps
    #
    #         model,y_pre = dbscan(data,j,i)
    #         s = silhouette_score(data, y_pre)
    #         result_label = model.labels_
    #         print(result_label)
    #         line.append(s)
    #     plt.plot(range(1,6),line,c=colors[i-1],label=('min_sam'+str(i)))
    # plt.ylabel("silhouette score")
    # plt.xlabel("eps")
    # plt.legend(('min_sam=2', 'min_sam=3','min_sam=4','min_sam=5','min_sam=6'),
    #            loc='upper left')
    # plt.show()

    pass