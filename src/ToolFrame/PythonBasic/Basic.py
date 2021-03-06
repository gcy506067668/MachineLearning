# -*- coding: UTF-8 -*-
import os

str = input("请输入：")
print("你输入的内容是: ", str)


#io相关
#   w+   覆盖式写入
#   a+   追加式写入
#   r+   读写 如果文件没有不会创建文件
file = open("test.txt","w+")
file.closed
file.mode
file.name
file.write("123")
file.close()

# 打开一个文件
fo = open("foo.txt", "r+")
str = fo.read(10)
# 查找当前位置
position = fo.tell()
# 把指针再次重新定位到文件开头
position = fo.seek(0, 0)
str = fo.read(10)
# 关闭打开的文件
fo.close()

#文件基本操作
os.rename("test.txt", "data.txt")
os.remove("test.txt")
os.makedirs("创建文件夹")
os.chdir("更改当前路径")
os.getcwd()   #显示当前路径
os.rmdir("删除路径")

#open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)

#异常捕获相关
try:
    fh = open("testfile", "w")
    fh.write("这是一个测试文件，用于测试异常!!")
except IOError:
    print("Error: 没有找到文件或读取文件失败")
else:
    print("内容写入文件成功")
    fh.close()

try:
    fh = open("testfile", "w")
    fh.write("这是一个测试文件，用于测试异常!!")
finally:
    print("无论有无错误，都会执行finally")


#列出所有文件
files = os.listdir("path_here")

#判断是否是文件夹
os.path.isdir("path")

#判断是否是文件
os.path.isfile("")


# list ['physics', 'chemistry', 1997, 2000]    下标索引

# tuple (1, 2, 3, 4, 5)    下标索引

# dict {'a': 1, 'b': 2, 'b': '3'}    key索引   exmple： dict['name']

#range()
# range(10) -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# range(1,10) -> [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# range(0,30,5) -> [0, 5, 10, 15, 20, 25]
# range(0,-10,-1) -> [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
