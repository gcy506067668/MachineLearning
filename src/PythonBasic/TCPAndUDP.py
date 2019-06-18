# -*- coding: UTF-8 -*-
import socket
import threading

import time


def tcpServer(port):
    server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind(('localhost',port))
    server.listen(5)#最大连接数 5
    conn,address=server.accept()
    while True:
        msg= conn.recv(1024)
        print(msg)
        conn.sendall(input("输入要发送的信息").encode('utf-8'))#以utf-8的字符集物转换成bytes
    conn.close()
    server.close()
    pass

def tcpClient(targetIp,port):
    client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client.connect((targetIp,port))
    while True:
        client.send(input("msg").encode('utf-8'))
        msg=client.recv(1024)
        print(msg.decode())
    client.close()

    pass


tcpClient("192.168.16.108",9000)

def udpSender():
    pass
def udpReceiver():
    pass

# threading.Thread(target=tcpServer,args=(10001,)).start()
# time.sleep(2)
# threading.Thread(target=tcpClient,args=("localhost",10001)).start()