import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import os
import ffmpeg
import numpy as np

video_path = "F:\\diskD\\test.mp4"
image_path = "./"

def avgColor(im):
    # im = Image.open(os.path.join('./test.jpg'))
    pix = im.load()
    width = im.size[0]
    height = im.size[1]
    r=[]
    g=[]
    b=[]
    for x in range(width):
        for y in range(height):
            rt, gt, bt = pix[x, y]
            r.append(rt)
            g.append(gt)
            b.append(bt)
    im.close()
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)
    red = np.mean(r)
    gree = np.mean(g)
    blue = np.mean(b)

    return (red.astype(int),gree.astype(int),blue.astype(int))
    # image = Image.new('RGB',(imw,imh),(255,255,255))
    # draw = ImageDraw.Draw(image)
    # for x in range(width):
    #     for y in range(height):
    #         draw.point((x, y), fill=(red.astype(int),gree.astype(int),blue.astype(int)))
    #
    # print(type(pix))
    # plt.figure("Image") # 图像窗口名称
    # plt.imshow(image)
    # plt.axis('on') # 关掉坐标轴为 off
    # plt.title('image') # 图像题目
    # plt.show()

def getImage(video_path, image_path):
    img_count = 1
    crop_time = 0.0
    while crop_time <= 6:#转化15s的视频
        # os.system('ffmpeg -i %s -f image2 -ss %s -vframes 1 %s.png'% (video_path, str(crop_time), image_path+ str(img_count)))
        print('ffmpeg -i %s -f image2 -ss %s -vframes 1 %s.png'% (video_path, str(crop_time), image_path+ str(img_count)))
        img_count += 1
        # print('Geting Image ' + str(img_count) + '.png' + ' from time ' + str(crop_time))
        crop_time += 0.1#每0.1秒截取一张照片
    print('视频转化完成！！！')

def connectImage():
    color = []
    for i in range(1,59):
        try:
            with Image.open('F:\\diskD\\test\\'+str(i)+'.png') as f:

                color.append(avgColor(f))
            pass
        except:
            pass
    return color

def image_to_txt(image_path, txt_path):
    # 这里使用到PIL库convert函数，将RGB图片转化为灰度图，参数'L'代表转化为灰度图
    im = Image.open(image_path).convert('L')
    charWidth = 100
    # 这个是设置你后面在cmd里面显示内容的窗口大小，请根据自己的情况，适当调整值
    im = im.resize((charWidth, charWidth // 2))
    target_width, target_height = im.size
    data = np.array(im)[:target_height, :target_width]
    f = open(txt_path, 'w',encoding='utf-8')
    for row in data:
        for pixel in row:
            if pixel > 127: # 如果灰度值大于127，也就是偏白的，就写一个字符 '1'
                f.write('1')
            else:
                f.write(' ')
        f.write('\n')
    f.close()
def getTxt(image_path, txt_path):#调用上面的函数image_to_txt
    img_count = 1# 一张图对应一个txt文件，所以每遍历一张图，该值加一

    while img_count <= len(os.listdir(image_path)):
        #os.listdir(image_path)# 返回所有图片名称，是个字符串列表
        imageFile = image_path+ str(img_count) + '.png'
        txtFile = txt_path+ str(img_count) + '.txt'
        image_to_txt.image_to_txt(imageFile, txtFile)
        print('舞蹈加载中： ' + str(img_count) + '%')
        img_count += 1


color = connectImage()
imw = len(color)
imh = 30

image = Image.new('RGB',(imw,imh),(255,255,255))
draw = ImageDraw.Draw(image)
index = 0

for i,item in enumerate(color):
    for j in range(30):
        draw.point((i, j), fill=item)
        # draw.point((i*5+1, j), fill=item)
        # draw.point((i*5+2, j), fill=item)
        # draw.point((i*5+3, j), fill=item)
        # draw.point((i*5+4, j), fill=item)

plt.figure("Image") # 图像窗口名称
plt.imshow(image)
plt.axis('on') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()

