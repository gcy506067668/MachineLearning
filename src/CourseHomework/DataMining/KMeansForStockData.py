# pip 镜像 https://pypi.tuna.tsinghua.edu.cn/simple
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import mpl_finance as mpf
from matplotlib.pylab import date2num
from dateutil.parser import parse


data_root_path = "./data"

def loadData(draw=False,count=100):
    if(draw):
        data = []
        for i in range(1,5):
            with open(data_root_path+"/stock_data_"+str(i)+".data") as f:
                while(True):
                    line_str = f.readline().replace("\n","").split(" ")
                    if(len(line_str) != 6):
                        break
                    results = list(map(float, line_str))
                    results[0] =  (date2num(parse(str(results[0]))))

                    data.append(tuple(results))
        pass
    else:
        data = []
        for i in range(1,5):
            with open(data_root_path+"/stock_data_"+str(i)+".data") as f:
                while(True):
                    line_str = f.readline().replace("\n","").split(" ")
                    if(len(line_str) != 6):
                        break
                    results = list(map(float, line_str))
                    #去除date
                    results.pop(0)
                    data.append(results)

        pass
    return data

#数据预处理
def preprocessing(data):
    # columnsData = ('open','close','high','low','volume')
    data_pd = pd.DataFrame(np.array(data), columns=list('01234'))
    data_mean = data_pd.mean()
    data_std = data_pd.std()
    data_zs = 1.0*(data_pd - data_mean)/data_std
    return data_zs,data_mean,data_std

def recoverPreProcessing(data,mean,std):
    data = np.array(data.values)
    std = np.array(std.values)
    mean = np.array(mean.values)
    data = data * std + mean
    return data


# k聚类数目   threadnum     iteration 迭代次数
def clustering(k,threadnum,iteration,data):
    model = KMeans(n_clusters = k, n_jobs = threadnum, max_iter = iteration)
    model.fit(data)
    return model

#data_list_ = [(date2num(parse(str(20181110))),10,20,5,15)]
# ##股票数据，格式是往列表里添加元组, 每个元组代表一个股票信息。
# 其中元组的格式是（日期，开盘价，最高价，最低价，收盘价）
def drawCandle(data):

    plt.rcParams['font.family'] = 'SimHei' ## 设置字体
    fig, ax = plt.subplots() ## 创建图片和坐标轴
    fig.subplots_adjust(bottom=0.2) ## 调整底部距离
    ax.xaxis_date() ## 设置X轴刻度为日期时间
    plt.xticks(rotation=45) ## 设置X轴刻度线并旋转45度
    plt.yticks() ## 设置Y轴刻度线
    plt.title("股票代码 ** K线图") ##设置图片标题
    plt.xlabel("时间") ##设置X轴标题
    plt.ylabel("股价（元）") ##设置Y轴标题
    plt.grid(True, 'major', 'both', ls='--', lw=.5, c='k', alpha=.3)  ##设置网格线

    mpf.candlestick_ohlc(ax,data,width=1.0,colorup='r',colordown='green', alpha=1)##设置利用mpf画股票K线图
    plt.show() ## 显示图片
    plt.savefig("K线.png") ## 保存图片
    plt.close() ## 关闭plt，释放内存

    pass


if __name__ =='__main__':
    data = loadData()
    data_zs,mean,std = preprocessing(data)
    kmModel = clustering(12,4,500,data_zs)

    result = pd.DataFrame(kmModel.cluster_centers_)
    result = recoverPreProcessing(result,mean,std)

    result = result.tolist()
    drawData = []
    for index,item in enumerate(result):
        temp = []
        if(index<9):
            date_d = '2019050'+str(index+1)
        else:
            date_d = '201905'+str(index+1)

        temp.append((date2num(parse(date_d))))
        temp.extend(item)

        drawData.append(tuple(temp))

    drawCandle(drawData)

