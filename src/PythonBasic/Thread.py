# -*- coding: UTF-8 -*-

import threading

def multiplePrint(text1,text2):
    print(text1)
    print(text2)
    pass

threading.Thread(target=multiplePrint,args=("11111","222222")).start()

