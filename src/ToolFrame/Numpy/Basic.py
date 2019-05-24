import numpy as np

# 1d
a = np.array([0, 1, 2, 3, 4])
b = np.array((0, 1, 2, 3, 4))
c = np.arange(5)
d = np.linspace(0, 2*np.pi, 5)  # 0 到 2Π 均匀分成5个元素
e = np.linspace(0, 4, 5)   # 0 到 4 均匀分成5个元素
e[2] = 10

# 2d
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])
# a[:,0] = 1
# a[0,:] = 1
# a[[0,2],:] = 0
# a[:,[0,2]] = 0
# a[0,:] = a[0,:] * 2
#print(a[::2,::2])      #此处可以理解为步长

# print(a.dtype) # >>>int64
# print(a.size) # >>>25
# print(a.shape) # >>>(5, 5)
# print(a.itemsize) # >>>8   每项占用字节数 此处int64
# print(a.ndim) # >>>2    数组的维数
# print(a.nbytes) # >>>200    数组中的所有数据消耗掉的字节数


np.prod((2,3,5))   #   计算2*3*5