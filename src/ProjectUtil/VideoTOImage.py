import cv2 as cv
import os

VIDEO_EXTENSIONS = ['.mp4', '.avi']
def isVideoFile(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


"""
    ：function:
        视频每隔n秒截取一帧图像并保存
    :parameter
        videoPath:           视频路径
        imgSavePath:         图片保存路径和图片名
        interval:            每隔多少秒截取一帧
    :return
        视频保存的总帧数
"""
def videoToImage(videoPath, imgSavePath, interval = 1):
    cap = cv.VideoCapture(videoPath)

    fps = cap.get(5)
    fps = int(fps)
    frame_count = cap.get(7)

    print("fps : "+str(fps))

    print("总贞数："+str(frame_count))

    ret = True
    imgIndex = 0

    while (ret):
        imgIndex += 1

        ret, frame = cap.read()

        if(imgIndex%(interval*fps)==0):
            cv.imwrite(imgSavePath+str(imgIndex)+".png",frame)
            print(imgSavePath+str(imgIndex)+".png")

    cap.release()
    return int(imgIndex/fps)
    pass


def cutVideosToImage(dataroot,img_save_path):
    if(os.path.isdir(dataroot)):

        for fileOrDir in os.listdir(dataroot):
            if os.path.isdir(dataroot+"/"+fileOrDir):
                cutVideosToImage(dataroot+"/"+fileOrDir,img_save_path+"/"+fileOrDir)


            elif isVideoFile(fileOrDir):
                # print(img_save_path+"/"+fileOrDir)
                if not os.path.exists(img_save_path):
                    os.makedirs(img_save_path)
                videoToImage(dataroot+"/"+fileOrDir,img_save_path+"/"+fileOrDir.replace(".mp4","_"))
        pass
    elif os.path.isfile(dataroot):
        if isVideoFile(dataroot):
            videoToImage(dataroot,img_save_path)
    return


if __name__ == '__main__':

    # command = os.popen("pwd")
    # dataroot = command.read()
    # command.close()
    dataroot = "/home/letmesleep/data/test"
    videos = cutVideosToImage(dataroot,dataroot+"_img")



