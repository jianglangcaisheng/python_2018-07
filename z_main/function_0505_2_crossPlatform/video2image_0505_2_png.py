
import datetime

import imageio
# imageio.plugins.ffmpeg.download()
import matplotlib.pyplot as plt
import numpy as np
import os


def mkdir(path):
    import os
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


# set
video_SID = "13_01"
# ...............................................................................................
pass
# nt: windows   posix: Linux
systemName = os.name
print("系统：" + systemName)
if systemName == "nt":
    path_system = "F:"
elif systemName == "posix":
    path_system = "/home/visg01/0jiangpeiyuan"

pathDir = path_system + "/0 SOG/0 data_video/br/" + video_SID
path_saveDir_root = path_system + "/0 SOG/0 data_video_image/br/" + video_SID

filename_common = "/MPI_br_" + video_SID + "_040_0"
filename_suffix = ".avi"

num_image = 0
begin = datetime.datetime.now()
for i_view in range(8):
    path_read = pathDir + filename_common + str(i_view) + filename_suffix
    path_saveDir = path_saveDir_root + filename_common + str(i_view) + ".png"
    mkdir(path_saveDir)

    vid = imageio.get_reader(path_read)

    # 100f/0.42s
    for num, im in enumerate(vid):
        num_image = np.maximum(num_image, num)
        # image = skimage.img_as_float(im).astype(np.float64)
        plt.imsave(path_saveDir + filename_common + str(i_view) + "_" + str(num) + ".png", im)
        if num % 100 == 0:
            print("\n" + '当前运行时间：' + str((datetime.datetime.now() - begin)))
            print("finish saving image: " + str(i_view)+ " " + str(num))
            print("finish video: " + video_SID)

num_image = num_image + 1
np.savez(path_saveDir_root + "/num_image.npz", num_image=num_image)