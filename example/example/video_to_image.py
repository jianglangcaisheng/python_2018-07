
import pylab
import imageio
import datetime
# 注释的代码执行一次就好，以后都会默认下载完成
# imageio.plugins.ffmpeg.download()
import skimage
import numpy as np


# example
if 0:
    filename = 'F:\\0 SOG\\0 data_video\\br\\14_01\\MPI_br_14_01_040_00.avi'
    vid = imageio.get_reader(filename)

    begin = datetime.datetime.now()
    # 100f/0.42s
    SID = 10
    for num, im in enumerate(vid):
        # image的类型是mageio.core.util.Image可用下面这一注释行转换为arrary
        # print(im.mean())
        if num==SID:
            print('当前运行时间：' + str((datetime.datetime.now() - begin)))


            image = skimage.img_as_float(im).astype(np.float64)
            # image = skimage.img_as_uint(im).astype(np.uint8)

            fig = pylab.figure()
            fig.suptitle('image #{}'.format(num), fontsize=20)
            pylab.imshow(image)     # image array 均可
            break
    pylab.show()


def getImage(path, SID):
    vid = imageio.get_reader(path)

    # begin = datetime.datetime.now()
    # 100f/0.42s
    for num, im in enumerate(vid):
        if num == SID:
            image_local = skimage.img_as_float(im).astype(np.float64)
    return image_local

def get8Image(path, SID):
    image_8view = []
    for i_view in range(1):
        path_view = path + str(i_view) + ".avi"
        image_8view.append(getImage(path_view, SID))
        print("get:" + str(i_view) + "_view")
    for i_view in range(1, 8):
        path_view = path + str(i_view) + ".avi"
        image_8view.append([])
        # print("get:" + str(i_view) + "_view")
    return image_8view


# exam_调用
if 1:
    path = 'F:\\0 SOG\\0 data_video\\br\\14_01\\MPI_br_14_01_040_00.avi'

    SID=0
    image = getImage(path=path, SID=SID)
    # fig = pylab.figure()
    # fig.suptitle('image #{}'.format(SID), fontsize=20)
    pylab.imshow(image)
    pylab.show()
