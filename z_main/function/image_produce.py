
import sys, os
import datetime

import numpy as np
import cv2

import scipy.io as sio
import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
import imageio
import skimage
from z_main.config_class import ModeConfig

import z_main.config as cf
import utility
import z_main.function_0505_2_crossPlatform.module_Lab as mLab
import z_main.function_mini as function_mini


def getImageFromPathfile(path_image):
    """return double"""

    Img = cv2.imread(path_image)
    if Img is None:
        print(path_image)
        assert False, "Img is None"
    Img_RGB = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    image_double = Img_RGB / 255.
    return image_double


def get8ImageFromImage(path, SID, views, postfix="bmp"):
    image_8view = []
    for i_view in views:
        if postfix=="bmp":
            path_view = path + "%d_bmp/%d-%d.%s" % (i_view, SID, i_view, postfix)
        elif postfix == "jpg":
            path_view = path + "%d.%s" % (i_view, postfix)
        else:
            assert False, "Error: " + postfix
        image_8view.append(getImageFromPathfile(path_view))
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
        # B.append([leftup[0], leftup[1], rightdown[0], rightdown[1]])
        B.append([leftup[1], leftup[0], rightdown[1], rightdown[0]])
        C.append(mean_color)
    elif sigma_color < eta:
        # B.append([leftup[0], leftup[1], rightdown[0], rightdown[1]])
        B.append([leftup[1], leftup[0], rightdown[1], rightdown[0]])
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


class CF_getBC:
    def __init__(self, leftup, rightdown, depth, path_video, i_frame, shape_or_pose):
        self.leftup = leftup
        self.rightdown = rightdown
        self.depth = depth
        self.path_video = path_video
        self.i_frame = i_frame
        self.shape_or_pose = shape_or_pose

# ger B, C      distributed
def get_B_C(i_view, cf_getBC):
    time_calcu = False

    begin = datetime.datetime.now()

    B = []
    C = []

    if cf_getBC.shape_or_pose == "sil_for_deep":
        postfix = "jpg"
    else:
        postfix = "bmp"

    image = get8ImageFromImage(path=cf_getBC.path_video, SID=cf_getBC.i_frame, views=[i_view], postfix=postfix)
    if time_calcu == True:
        print("\ni_frame: %d" % cf_getBC.i_frame)
        print("Time of getting 1 image: " + str((datetime.datetime.now() - begin)))

    cluster_quad(im=image[0], leftup=cf_getBC.leftup, rightdown=cf_getBC.rightdown,
                 depth=cf_getBC.depth, B=B, C=C)
    if time_calcu == True:
        print("Time of clustering 1 image: " + str((datetime.datetime.now() - begin)))

    # write to file
    if 1:
        pathSave_imageClustered = cf.pathVideo_ + "v%d/" % i_view
        utility.mkdir(pathSave_imageClustered)

        file = open(pathSave_imageClustered + "image%d_B.txt" % cf_getBC.i_frame, 'w')
        for fp in B:
            file.write(str(fp))
            file.write('\n')
        file.close()

        file = open(pathSave_imageClustered + "image%d_C.txt" % cf_getBC.i_frame, 'w')
        for fp in C:
            file.write(str(fp))
            file.write('\n')
        file.close()
    if time_calcu == True:
        print("Time of saving 1 cluster of image: " + str((datetime.datetime.now() - begin)))

    # return [B, C]
    return


def getImageProduced(mu_sigma_D_EII, image_color_clustered, max_of_cluster, mode_config):

    B_mu_new_fs = mu_sigma_D_EII[0]
    B_sigma_new_fs = mu_sigma_D_EII[1]
    D_new_fs = mu_sigma_D_EII[2]
    EII_new_fs = mu_sigma_D_EII[3]
    ifAImage = mode_config.ifAImage
    ifShape = mode_config.shape_or_pose
    mode = mode_config.mode
    path_Tpose = mode_config.path_Tpose
    path_TposeColor = mode_config.path_TposeColor
    getImageForColor = image_color_clustered[0]
    getImageClustered = image_color_clustered[1]

    B_video = []
    D_video = []
    EII_video = []

    B_video.clear()
    D_video.clear()
    EII_video.clear()

    B8 = []
    C8 = []
    D8 = []
    EII8 = []

    if ifAImage == 1:
        silOrColor = "sil"
    else:
        silOrColor = "col"

    # get B, C, D, EII
    if 1:
        class ModeConfig_prePropose:
            def __init__(self, mode, path_Tpose, ifShape, path_TposeColor, silOrColor):
                self.Num_point = 63
                self.epsilon = 0.15
                self.mode = mode
                self.path_Tpose = path_Tpose,
                self.ifShape = ifShape
                self.path_TposeColor = path_TposeColor
                self.silOrColor = silOrColor

        mode_config_prePropose = ModeConfig_prePropose(mode, path_Tpose, ifShape, path_TposeColor, silOrColor)

        def get_B(B8_C8, max_of_cluster, mode_config_prePropose, mode_config):

            mode = mode_config_prePropose.mode

            depth = 6
            if mode == "sil_for_deep":
                depth = depth + 2
            else:
                depth = depth

            num_cluster_row = np.zeros([1, 8])



            B8, C8 = B8_C8


            cf_getBC = CF_getBC(leftup=[0, 0], rightdown=[cf.imageHeight, cf.imageWidth], depth=depth,
                                path_video=mode_config.path_video, i_frame=mode_config.i_frame, shape_or_pose=mode)

            for i_view in range(8):
                begin = datetime.datetime.now()

                pathSave_imageClustered = cf.pathVideo_ + "v%d/" % i_view
                if (not os.path.exists(pathSave_imageClustered + "image%d_B.txt" % cf_getBC.i_frame)) or 1:
                # if (not os.path.exists(pathSave_imageClustered + "image%d_B.txt" % cf_getBC.i_frame)) or cf_getBC.i_frame < 150:
                    get_B_C(i_view, cf_getBC)
                    print("get_B_C")
                else:
                    print("Use B_C before.")

                # print(pathSave_imageClustered + "image%d_B.txt" % cf_getBC.i_frame)

                B = []
                C = []
                B.clear()
                C.clear()

                begin_read_image = datetime.datetime.now()
                file = open(pathSave_imageClustered + "image%d_B.txt" % cf_getBC.i_frame, 'r')
                for i_row in file:
                    i_row_str = i_row.lstrip('[').rstrip(']\n').split(', ')
                    i_row_number = [int(x) for x in i_row_str]
                    try:
                        B.append(i_row_number)
                    except MemoryError:
                        print(B.shape)
                file.close()

                file = open(pathSave_imageClustered + "image%d_C.txt" % cf_getBC.i_frame, 'r')
                from functools import reduce
                def str2float(s):
                    L = s.rstrip().split('.')
                    if L[1] == "":
                        L[1] = "0"
                    return reduce(lambda x, y: y + x * 10, map(int, L[0])) + reduce(lambda x, y: y + x * 10,
                                                                                 map(int, L[1])) / 10 ** len(L[1])

                for i_row in file:
                    i_row_str = i_row.lstrip('[').rstrip(']\n').split()
                    i_row_number = [str2float(x) for x in i_row_str]
                    try:
                        C.append(i_row_number)
                    except:
                        dlt = 0
                file.close()

                if 0:
                    print("Time of reading 1 cluster of image: " + str((datetime.datetime.now() - begin_read_image)))

                B8.append(B)
                C8.append(C)
                num_cluster = C.__len__()
                num_cluster_row[0][i_view] = num_cluster
                if 0:
                    print("view: %d, num_cluster: %d. " % (i_view, num_cluster), end="")
                    print("Time of producing 1 images to get B and C: " + str((datetime.datetime.now() - begin)))

            B8, C8 = B8_C8

            # save
            if os.path.exists(cf.pathFile_numCluster):
                data = sio.loadmat(cf.pathFile_numCluster)
                num_cluster_last = data['num_cluster']
                max_of_cluster_last = data['max_of_cluster']

                num_cluster = np.concatenate([num_cluster_last, num_cluster_row], 0)
                max_of_cluster = max(max_of_cluster_last, max_of_cluster)

                sio.savemat(cf.pathFile_numCluster, {'num_cluster': num_cluster,
                                                     'max_of_cluster': max_of_cluster})
            else:
                if max_of_cluster == None:
                    sio.savemat(cf.pathFile_numCluster, {'num_cluster': num_cluster_row})
                    print(utility.str2red("max_of_cluster is None"))
                else:
                    sio.savemat(cf.pathFile_numCluster, {'num_cluster': num_cluster_row,
                                                         'max_of_cluster': max_of_cluster})

            # prepare D EII
            body = read_Tpose(path_Tpose=path_Tpose)
            if path_TposeColor != "":
                body = updateBodyByColor(body, path_bodyColor=path_TposeColor)

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

                # todo 颜色重写
                LAB_space = mLab.RGB2LAB(RGB_space_uint8)
                LAB_space[:, :, 0] = function_mini.compress_L_in_Lab(LAB_space[:, :, 0])
                RGB_space_new = mLab.LAB2RGB(LAB_space)
                color_all = RGB_space_new / 255.


            return color_all

        B8_C8 = [B8, C8]
        color_all = get_B(B8_C8, max_of_cluster, mode_config_prePropose, mode_config)

        def calcu_D_EII(D8_EII8):
            [D8, EII8] = D8_EII8
            # D
            for i_view in range(8):
                # D = []
                color_all_1_3000_3 = np.transpose(color_all[:, i_view:i_view + 1, :], [1, 0, 2])
                color_im = np.broadcast_to(color_all_1_3000_3,
                                           [mode_config_prePropose.Num_point, cf.image_cluster_number, 3])
                color_point = np.broadcast_to(color_all[0:mode_config_prePropose.Num_point, 8:9, :],
                                              [mode_config_prePropose.Num_point, cf.image_cluster_number, 3])

                try:
                    s = np.sqrt(
                        np.sum((color_im - color_point) * (color_im - color_point), axis=2)
                    ) / mode_config_prePropose.epsilon
                except MemoryError:
                    print(color_im.shape)
                    raise MemoryError

                # D_temp: shape: [63, 24000], average: 0.95
                # penalty   -0.1
                if ifShape == True:
                    D_temp = np.ones([mode_config_prePropose.Num_point, cf.image_cluster_number]) * (-0.3)
                elif silOrColor == "sil":
                    D_temp = np.ones([mode_config_prePropose.Num_point, cf.image_cluster_number]) * (-0.3)
                else:
                    D_temp = np.zeros([mode_config_prePropose.Num_point, cf.image_cluster_number])
                D_temp[s < 1] = np.power((1 - s[s < 1]), 4) * (4 * s[s < 1] + 1)

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

        D8_EII8 = [D8, EII8]
        calcu_D_EII(D8_EII8)

    # imshow cluster
    if 1:
        for i_num_view in range(cf.num_view):
            image = np.zeros((1536, 2048, 3), dtype="uint8") * 255
            print("view: %d, num_cluster: %d" % (i_num_view, B8[i_num_view].__len__()))
            for i_ractange in range(B8[i_num_view].__len__()):

                color_0 = C8[i_num_view][i_ractange][2]
                color_1 = C8[i_num_view][i_ractange][1]
                color_2 = C8[i_num_view][i_ractange][0]
                color = np.array([color_0, color_1, color_2]) * 255

                try:
                    cv2.rectangle(image, (B8[i_num_view][i_ractange][0], B8[i_num_view][i_ractange][1]),
                                  (B8[i_num_view][i_ractange][2], B8[i_num_view][i_ractange][3]),
                                  color, -1)
                except:
                    print("Maybe: file error receiving.")
                    print((B8[i_num_view][i_ractange][0], B8[i_num_view][i_ractange][1]))
                    assert False, "List exceeds. View: %d" % i_num_view

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


