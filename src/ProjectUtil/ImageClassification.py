import os
import cv2
import shutil
"""
function:
    手工给图片分类
    使用方法：运行之后确认获得图片窗口焦点，键盘输入0给当前图片分入savepath目录下的0文件夹，
            输入1给当前图片分入savepath下1文件夹...
parameters:
    dataroot:需要分类的图片目录，程序会自动搜索该目录下所有图片文件
    savepath:分类保存的文件夹目录
    classcount:图片的类别数
    tip：窗口提示内容，比如可以在窗口上方提示数字对应的类别
    
    
image classification util
create by letmesleep
email:506067668@qq.com
"""
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def findAllImages(dataroot):
    images = []
    if (os.path.isdir(dataroot)):

        for fileOrDir in os.listdir(dataroot):
            if os.path.isdir(os.path.join(dataroot, fileOrDir)):
                findAllImages(os.path.join(dataroot, fileOrDir))

            elif any(fileOrDir.endswith(extension) for extension in IMG_EXTENSIONS):
                images.append(os.path.join(dataroot, fileOrDir))
        pass
    
    return images


def Imgclassification(dataroot, savepath, classcount=2, tips="image"):
    images = findAllImages(dataroot)
    index = 0
    imagelength = len(images)
    print("total images: "+str(imagelength))
    for image in images:

        cv2.imshow(tips, cv2.imread(image))
        print("current image : "+image)
        print("num :"+str(index)+"/"+str(imagelength))

        saveFileFlag = False

        while True:
            key = cv2.waitKey(0)
            for cla in range(classcount):
                if key == ord(str(cla)):
                    if not os.path.isdir(os.path.join(savepath, str(cla))):
                        os.makedirs(os.path.join(savepath, str(cla)))

                    shutil.copyfile(image, os.path.join(savepath, str(cla), image.split(os.path.sep)[-1]))
                    saveFileFlag = True
                    break



            if saveFileFlag:
                break
    return

if __name__ == '__main__':

    dataroot = "/media/letmesleep/My Passport/处理后的XG/QJ/3_normal"
    savepath = "/media/letmesleep/My Passport/处理后的XG/QJ/QJ_classification"
    classcount = 4
    tips = "0-normal  1-abnormal  2-other  3-delete"
    Imgclassification(dataroot, savepath, classcount, tips)