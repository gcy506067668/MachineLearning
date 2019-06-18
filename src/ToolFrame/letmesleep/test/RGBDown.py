import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import os

def showRGBDown():
    im = Image.open(os.path.join('./test.jpg'))
    plt.imshow(im)
    plt.show()
    pix = im.load()
    width = im.size[0]
    height = im.size[1]


    image = Image.new('RGB', (width*5, height*5), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for times in range(25):
        for x in range(width):
            for y in range(height):
                rt, gt, bt = pix[x, y]
                r,g,b = 0,0,0
                if(rt-10*times<0):
                    r=0
                else:
                    r=(rt-10*times)
                if(gt-10*times<0):
                    g=(0)
                else:
                    g=(rt-10*times)

                if(bt-10*times<0):
                    b=(0)
                else:
                    b=(rt-10*times)


                draw.point((x+width*(times%5), y+(int)(times/5)*height), fill=(r,g,b))

    im.close()



    plt.figure("Image")  # 图像窗口名称
    plt.imshow(image)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.show()

def show3DImage():
    im = Image.open(os.path.join('./test.jpg'))

    fig = plt.figure()
    ax = fig.gca(projection='3d')


    pix = im.load()
    width = im.size[0]
    height = im.size[1]

    points_x = []
    points_y = []
    points_z = []
    c_list = []
    for x in range(width):
        for y in range(height):
            rt, gt, bt = pix[x, y]
            points_x.append(x)
            points_y.append(y)
            points_z.append((int)((rt+gt+bt)/3))
            c_list.append(((rt/255),(gt/255),(bt/255)))
    # Plot scatterplot data (20 2D points per colour) on the x and z axes.



    # By using zdir='y', the y value of these points is fixed to the zs value 0
    # and the (x,y) points are plotted on the x and z axes.
    ax.scatter(points_x, points_y, points_z, c=c_list, label='points in (x,y,z)')

    # Make legend, set axes limits and labels
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=50., azim=80)

    plt.show()

<<<<<<< HEAD
showRGBDown()
=======
show3DImage()
>>>>>>> cdbeead0c47804e84edc5c263f36b2f3f8793492
