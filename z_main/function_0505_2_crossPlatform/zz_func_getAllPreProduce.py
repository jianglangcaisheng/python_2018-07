
import sys, os
import datetime

import numpy as np
import cv2

import scipy.io as sio

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
import imageio
import skimage

import z_main.config as cf
import z_main.function_0505_2_crossPlatform.module_Lab as mLab
import z_main.function_mini as function_mini


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


def getImage(path, SID):
    vid = imageio.get_reader(path)

    # begin = datetime.datetime.now()
    # 100f/0.42s
    for num, im in enumerate(vid):
        if num == SID:
            image_local = skimage.img_as_float(im).astype(np.float64)
    return image_local


def getImageFromImage(path, SID):
    """return double"""
    # begin = datetime.datetime.now()
    # 100f/0.42s
    path_image = path
    Img = cv2.imread(path_image)
    if Img is None:
        print(path_image)
        print("Img is None")
    # if(Img.any() == None):
    #     print("Img None")

    # BGR   requested: uint8
    Img_RGB = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

    image_double = Img_RGB / 255.


    # cv2.namedWindow("image_double")
    # cv2.imshow("image_double", image_double)
    # cv2.waitKey(0)
    return image_double


def getImageFromAImage(path):
    """return double"""
    # begin = datetime.datetime.now()
    # 100f/0.42s
    path_image = path

    Img = cv2.imread(path_image)
    if Img is None:
        print(path_image)
        assert Img, "Img is None"

    # windowName = "tmp"
    # cv2.namedWindow(windowName, 0)
    # cv2.resizeWindow(windowName, 1024, 768)
    # cv2.imshow(windowName, Img)
    # cv2.waitKey(0)

    # BGR   requested: uint8
    Img_RGB = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

    image_double = Img_RGB / 255.


    # cv2.namedWindow("image_double")
    # cv2.imshow("image_double", image_double)
    # cv2.waitKey(0)
    return image_double


def get8Image(path, SID):
    image_8view = []
    for i_view in range(8):
        path_view = path + str(i_view) + ".avi"
        image_8view.append(getImage(path_view, SID))
        # print("get:" + str(i_view) + "_view")
    # for i_view in range(1, 8):
    #     path_view = path + str(i_view) + ".avi"
    #     image_8view.append([])
    #     # print("get:" + str(i_view) + "_view")
    return image_8view


def get8ImageFromImage(path, SID, video_choose):
    image_8view = []
    for i_view in range(8):
        str_view = "%d/%d-%d.bmp" % (i_view, SID, i_view)
        path_view = path + str_view
        image_8view.append(getImageFromImage(path_view, SID))
        # print("get:" + str(i_view) + "_view")
    return image_8view


def get8ImageFromAImage(path, mode=None, modeConfig = None):
    image_8view = []

    for i_view in range(8):
        str_view = str(i_view)


        if mode == "optimize_shape_inRGB":
            path_view = path + "%d/%d-%d.bmp" % (i_view, modeConfig['i_image'] - modeConfig['id'] * cf.shape_delta, i_view)
        else:
            # 剪影不能png，全0
            # 图片不能bmp，全0
            # 转jpg！！！
            path_view = path + str_view + ".jpg"
        image_8view.append(getImageFromAImage(path_view))

    return image_8view


def read_Tpose(path_Tpose):
    data = sio.loadmat(path_Tpose)
    body = data['body']
    return body


def updateBodyByColor(body, path_bodyColor):
    data = sio.loadmat(path_bodyColor)
    # todo 查询
    bodyColor = data['color']
    # bodyColor = data['bodyColor']
    body[0][2] = bodyColor
    return body


def cluster_quad(im,leftup,rightdown,depth, B, C):
    # global num_cluster
    # num_cluster += 1
    # print(leftup)
    # print(rightdown)
    # print(str(num_cluster) + '\n')
    # todo x.y  1004,
    image_shape = im.shape[0]   # b
    eta = 0.03

    imblob = im[leftup[0]:rightdown[0], leftup[1]:rightdown[1], :]
    [imblob_x, imblob_y, _] = imblob.shape
    im_color = np.reshape(imblob, [imblob_x * imblob_y, 3])

    sigma_color = np.linalg.norm(np.std(a=im_color, ddof=1, axis=0))
    mean_color = np.mean(a=im_color, axis=0)

    mu = np.int32([np.floor((leftup[0] + rightdown[0]) / 2),
                   np.floor((leftup[1] + rightdown[1]) / 2)])

    s = depth

    if image_shape / (pow(2, s)) <= 1:
        s = 4

    if(-leftup[1] + rightdown[1]) < image_shape / (pow(2, s)):
        B.append([leftup[0], leftup[1], rightdown[0], rightdown[1]])
        C.append(mean_color)
    elif sigma_color < eta:
        B.append([leftup[0], leftup[1], rightdown[0], rightdown[1]])
        C.append(mean_color)
    else:
        cluster_quad(im=im,
                     leftup=leftup,
                     rightdown=mu, depth=depth, B=B, C=C)
        cluster_quad(im=im,
                     leftup=[leftup[0], mu[1]],
                     rightdown=[mu[0], rightdown[1]], depth=depth, B=B, C=C)
        cluster_quad(im=im,
                     leftup=[mu[0], leftup[1]],
                     rightdown=[rightdown[0], mu[1]], depth=depth, B=B, C=C)
        cluster_quad(im=im,
                     leftup=mu,
                     rightdown=rightdown, depth=depth, B=B, C=C)

    return


def prePropose(image_8view, B8, C8, D8, EII8, path_Tpose, ifShape=False, path_TposeColor="", silOrColor="sil", mode=None, max_of_cluster=None):

    depth = 6
    if mode == "sil_for_deep":
        depth = depth + 2
    elif mode == "optimize_shape_inRGB":
        depth = depth
    else:
        depth = depth

    Num_point = 63
    epsilon = 0.15
    # num_cluster = np.zeros(8)
    if max_of_cluster is not None:
        print("Num of cluster: ", end="")

    num_cluster_row = np.zeros([1, 8])
    for i_view in range(8):
        B = []
        C = []

        cluster_quad(im=image_8view[i_view], leftup=[0, 0], rightdown=[cf.imageHeight, cf.imageWidth],
                     depth=depth, B=B, C=C)
        B_yx = []
        # todo speed
        for i_B in range(B.__len__()):
            B_yx.append([B[i_B][1], B[i_B][0], B[i_B][3], B[i_B][2]])
        B8.append(B_yx)
        C8.append(C)

        num_cluster = C.__len__()
        num_cluster_row[0][i_view] = num_cluster
        if max_of_cluster is not None:
            if num_cluster > max_of_cluster:
                max_of_cluster = num_cluster
        print("%d, " % num_cluster, end="")
    if max_of_cluster is None:
        max_of_cluster = 0
    print("    max: %d" % max_of_cluster)

    if os.path.exists(cf.pathFile_numCluster):
        data = sio.loadmat(cf.pathFile_numCluster)
        num_cluster_last = data['num_cluster']
        max_of_cluster_last = data['max_of_cluster']

        num_cluster = np.concatenate([num_cluster_last, num_cluster_row], 0)
        max_of_cluster = max(max_of_cluster_last, max_of_cluster)

        sio.savemat(cf.pathFile_numCluster, {'num_cluster': num_cluster,
                                             'max_of_cluster': max_of_cluster})
    else:
        sio.savemat(cf.pathFile_numCluster, {'num_cluster': num_cluster_row,
                                             'max_of_cluster': max_of_cluster})

    body = read_Tpose(path_Tpose=path_Tpose)
    if path_TposeColor != "":
        body = updateBodyByColor(body, path_bodyColor = path_TposeColor)

    RGB_space = np.zeros([cf.image_cluster_number, 9, 3])
    for i_view in range(8):
        for i_C in range(C8[i_view].__len__()):
            # 挂载第三维. todo: 统计
            RGB_space[i_C, i_view, :] = np.array([C8[i_view][i_C][0], C8[i_view][i_C][1], C8[i_view][i_C][2]])

    # float64
    RGB_space[0:63, 8, :] = body[0][2]

    # color transform
    if silOrColor == "sil":
        color_all = RGB_space
    elif silOrColor == "col":
        RGB_space_uint8 = np.uint8(RGB_space * 255)

        # dlt
        if 0:
            image_dlt = image_8view[0]
            image_dlt_uint8 = np.uint8(image_dlt * 255)
            image_dlt_look = image_dlt_uint8[:, :, 1]


            image_Lab_dlt = cv2.cvtColor(image_dlt_uint8, cv2.COLOR_RGB2LAB)
            image_Lab_dlt_look_L = image_Lab_dlt[:, :, 0]
            image_Lab_dlt_look_a = image_Lab_dlt[:, :, 1]
            image_Lab_dlt_look_b = image_Lab_dlt[:, :, 2]

            image_New_dlt = cv2.cvtColor(image_Lab_dlt, cv2.COLOR_LAB2RGB)
            image_New_dlt_look = image_New_dlt[:, :, 1]

        # todo 颜色重写
        LAB_space = mLab.RGB2LAB(RGB_space_uint8)
        # LAB_space[:, :, 0] = cf.L_in_LabSpace                                                               # warning
        LAB_space[:, :, 0] = function_mini.compress_L_in_Lab(LAB_space[:, :, 0])
        RGB_space_new = mLab.LAB2RGB(LAB_space)
        color_all = RGB_space_new / 255.

    # D
    for i_view in range(8):
        # D = []
        color_all_1_3000_3 = np.transpose(color_all[:, i_view:i_view + 1, :], [1, 0, 2])
        color_im = np.broadcast_to(color_all_1_3000_3, [Num_point, cf.image_cluster_number, 3])
        color_point = np.broadcast_to(color_all[0:Num_point, 8:9, :], [Num_point, cf.image_cluster_number, 3])

        s = np.sqrt(
                np.sum((color_im - color_point ) * (color_im - color_point ), axis=2)
            ) / epsilon

        # D_temp: shape: [63, 24000], average: 0.95
        # penalty   -0.1
        if ifShape == True:
            D_temp = np.ones([Num_point, cf.image_cluster_number]) * (-0.3)
        elif silOrColor == "sil":
            D_temp = np.ones([Num_point, cf.image_cluster_number]) * (-0.3)
        else:
            D_temp = np.zeros([Num_point, cf.image_cluster_number])
        D_temp[s<1] = np.power((1 - s[s<1]), 4) * (4 * s[s<1] + 1)

        D8.append(D_temp)

        # todo label减少聚类数量

    # todo EII
    for i_view in range(8):
        B_array_temp_in_EII = np.zeros([cf.image_cluster_number, 4])
        for i_cluster in range(B8[i_view].__len__()):
            B_array_temp_in_EII[i_cluster, :] = B8[i_view][i_cluster]
        sigma_im = np.floor(0.5 * (B_array_temp_in_EII[:, 2:3] - B_array_temp_in_EII[:, 0:1]))
        EII = np.pi * (np.transpose(sigma_im, [1, 0]) * np.transpose(sigma_im, [1, 0]))
        EII8.append(EII)


# main  ############################################################################################


def getImageProduced(B_mu_new_fs, B_sigma_new_fs, D_new_fs, EII_new_fs, i_frame, path_video, path_Tpose, video_choose,
                     ifAImage=0, ifShape=False, getImageForColor=None, mode=None,
                     path_video_forColor="", path_TposeColor="", getImageClustered=None, max_of_cluster=None, modeConfig=None):
    B_video = []
    D_video = []
    EII_video = []

    B_video.clear()
    D_video.clear()
    EII_video.clear()

    if ifAImage == 1:
        image_8view = get8ImageFromAImage(path=path_video, mode=mode, modeConfig=modeConfig)
        if ifShape == True and (mode != "optimize_shape_inRGB"):
            image_8view_forColor = get8ImageFromAImage(path=path_video_forColor)
    else:
        image_8view = get8ImageFromImage(path=path_video, SID=i_frame, video_choose=video_choose)
        # image_8view_forColor = get8ImageFromImage(path=path_video_forColor, SID=SID, video_choose=video_choose)

    if ifShape == True and (mode != "optimize_shape_inRGB"):
        getImageForColor.append(image_8view_forColor)

    B8 = []
    C8 = []
    D8 = []
    EII8 = []

    if ifAImage == 1:
        silOrColor = "sil"
    else:
        silOrColor = "col"
    prePropose(image_8view, B8, C8, D8, EII8, path_Tpose=path_Tpose,
               ifShape=ifShape, path_TposeColor=path_TposeColor, silOrColor=silOrColor, mode=mode, max_of_cluster=max_of_cluster)

    # imshow cluster
    if 1:
        for i_num_view in range(cf.num_view):
            image = np.zeros((1536, 2048, 3), dtype="uint8") * 255
            for i_ractange in range(B8[i_num_view].__len__()):
                # print(str(i_ractange) + ": " + str(B8[0][i_ractange][0]) + "  " + str(B8[0][i_ractange][1]))
                # print("\n" + str(i_num_view) + "_" + str(i_ractange))
                # print(C8[i_num_view][i_ractange] * 255)

                color_0 = C8[i_num_view][i_ractange][2]
                color_1 = C8[i_num_view][i_ractange][1]
                color_2 = C8[i_num_view][i_ractange][0]
                color = np.array([color_0, color_1, color_2]) * 255
                cv2.rectangle(image, (B8[i_num_view][i_ractange][0], B8[i_num_view][i_ractange][1]),
                              (B8[i_num_view][i_ractange][2], B8[i_num_view][i_ractange][3]),
                              color, -1)

            if getImageClustered is None:
                pass
            else:
                getImageClustered.append(image)


    B_video.append(B8)
    D_video.append(D8)
    EII_video.append(EII8)

    # prepro_pose
    B_local = B_video
    D_local = D_video
    EII_local = EII_video

    for i_frame in range(B_local.__len__()):
        B_mu_ave = np.zeros([8, cf.image_cluster_number, 2])
        B_sigma_ave = np.zeros([8, cf.image_cluster_number])
        D_sum_ave = np.zeros([8, 63, cf.image_cluster_number])
        EII_sum_ave = np.zeros([8, cf.image_cluster_number])

        # todo matlab uint16 to float64
        for i_view in range(8):
            # copy
            size_of_cluster_local = np.size(B_local[i_frame][i_view], 0)

            B_local_view = np.float64(B_local[i_frame][i_view])
            D_local_view = np.float64(D_local[i_frame][i_view])
            EII_local_view = np.float64(EII_local[i_frame][i_view])

            B_mu_ave[i_view, 0:size_of_cluster_local, 0] = (B_local_view[:, 0] + B_local_view[:, 2]) / 2
            B_mu_ave[i_view, 0:size_of_cluster_local, 1] = (B_local_view[:, 1] + B_local_view[:, 3]) / 2
            B_sigma_ave[i_view, 0:size_of_cluster_local] = (abs(B_local_view[:, 0] - B_local_view[:, 2]) +
                                                        abs(B_local_view[:, 1] - B_local_view[:, 3])) / 4

            D_sum_ave[i_view, :, :] = D_local_view[:, :]
            EII_sum_ave[i_view, :] = EII_local_view[:, :]

        # transform size
        B_mu_new = np.transpose(B_mu_ave, [1, 0, 2])
        B_sigma_new = np.transpose(B_sigma_ave, [1, 0])
        D_new = np.transpose(D_sum_ave, [1, 2, 0])
        EII_new = np.transpose(EII_sum_ave, [1, 0])

        B_mu_new_fs.append(B_mu_new)
        B_sigma_new_fs.append(B_sigma_new)
        D_new_fs.append(D_new)
        EII_new_fs.append(EII_new)



