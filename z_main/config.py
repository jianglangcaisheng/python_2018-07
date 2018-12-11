

import numpy as np
import scipy.io as sio
import datetime
import math
import os
import utility
import shutil

# todo small:
# image_result
# string %
# timei_posei
# log
# draw shape 2
# graph structure
# run para

# todo structure:
# congif core
pass

#  system config.........................................................................................................
if 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Params........................................................................................................................
if 1:
    CameraSystem_L = ["PC1", "PC2"]
    CameraSystem_S = ["s0", "s1"]
    if 0:
        arg = input("PC: ")
    else:
        matlab_data = sio.loadmat(r"F:\0 SoG\0 SOG_201807\0 SOG_201807/PC.mat")
        arg = matlab_data['PC']
        print("PC: %d" % arg)
        arg = str(arg[0][0])

    if arg == "0":
        PC = CameraSystem_L[0]
    elif arg == "1":
        PC = CameraSystem_L[1]
    elif arg == "2":
        PC = CameraSystem_S[0]
    else:
        assert False, "Error arg"

    trainStepPose = 0.00000001 * 3 / 10 / 10 / 1

    num_times_in_shape = 4

    # 1200
    trainTimes_poseInShape_array = [1800, 1500, 1200, 1200, 1200, 1200]
    timePer_trainTimes_poseInShape = 55 / 500
    # trainTimes_shapeInShape = 1800

    # 4400 + 1800
    trainTimes_shapeInShape_array = [2200, 900, 900, 900, 900, 900]
    timePer_trainTimes_shapeInShape = 17 / 50

    for i_num_times_in_shape in range(num_times_in_shape - 6):
        trainTimes_poseInShape_array.append(1200)
        trainTimes_shapeInShape_array.append(900)

    # trainTimes_shapeCoefficient = 500
    if 1:
        trainTimes_poseInPose = int(1000 * 1.5)
    else:
        trainTimes_poseInPose = 1

    idImage = (150, 150)                                                                                # warning1:new_data

    imageWidth = 2048
    imageHeight = 1536
    image_cluster_number = int(24000)
    # image_cluster_number = 3000
    bodyNumBall = 63
    num_view = 8

# path..................................................................................................................
if 1:
    class PathImage:
        def __init__(self, PC):
            pathHalf_image = "0_image_1811_2_cali/K1_RT1"  # warning1:new_data
            if PC == "Desktop":
                path_image_half = r"X:\0_project_SoG\0_dataset/" + pathHalf_image
            else:
                path_image_half = r"D:\\" + pathHalf_image

            pathCameraParams = path_image_half + "/" + "data_detect/M_fl.mat"
            pathVideo_ = path_image_half + "_D1/"  # warning1:new_data

    class PathManager:
        def __init__(self, PC):
            path_project = None
            path_image = PathImage(PC)

    path_manager = PathManager(PC)

if 1:
    if 1:
        if 0:
            # nt: windows   posix: Linux
            systemName = os.name
            print("系统：" + systemName)
            if systemName == "nt":
                path_system = "J:"
            elif systemName == "posix":
                path_system = "/home/visg01/0jiangpeiyuan"

        # path_dir = path_system + "/0 SOG/0 data"

        if PC == "Desktop":
            path_data = "J:/0 SOG_201807/data/"
        elif PC in CameraSystem_L:
            path_data = "F:/0 SoG/data/"
        elif PC in CameraSystem_S:
            path_data = "C:/0_SoG/data/"

    # Path Save
    if 1:
        # level 1
        str_timeDate_level_1 = "181128"
        str_dir_level_1 = 'dataResult_%s_image1811_1_K1/' % (str_timeDate_level_1)
        str_dir_level_1 = 'dataResult_%s_image1811_2_K1/' % (str_timeDate_level_1)
        pathSave = path_data + str_dir_level_1
        if PC in CameraSystem_L:
            utility.mkdir(pathSave)

        str_level_2_list = [
            ["181114", "0018", "D116"],
            ["181120", "1959", "D115"],
            ["181123", "2139", "D115_newSil"],
            ["181123", "2333", "D115_newSil_fixImageProcessed"],
            ["181124", "0021", "D114"],
            ["181125", "1927", "D114_root16"],
            ["181125", "2130", "D113_root16"],
            ["181126", "2152", "D113_root16_weight2"],
            ["181126", "2152", "D112_root16_weight2"],
            ["181128", "0310", "D111_root16_weight2"],
            ["181211", "1731", "D111_test"]
        ]

        str_level_2 = str_level_2_list[-1]
        str_timeDate_level_2 = str_level_2[0]
        str_dir_final = "%s_%s_%s/" % (str_level_2[0], str_level_2[1], str_level_2[2])

        str_DateTime_save = str_timeDate_level_2

        path_save_direction = pathSave + str_dir_final
        if PC in CameraSystem_L:
            utility.mkdir(path_save_direction)

        # copy last data
        if 1 and PC in CameraSystem_L:
            print("Copy files......")
            str_level_2_last = str_level_2_list[-2]
            str_dir_final_last = "%s_%s_%s/" % (str_level_2_last[0], str_level_2_last[1], str_level_2_last[2])

            if 1:
                str_DateTime_save_last = "181124"
                str_DateTime_save_last = str_level_2_last[0]
                files_tobeCopied = [
                    str_DateTime_save_last + "_3_shape.mat",
                    str_DateTime_save_last + "_3_color_pose0.mat",
                    str_DateTime_save_last + "_pose_285.mat"
                ]
            else:
                files_tobeCopied = [
                    "181023_3_shape.mat",
                    "181023_3_color_pose0.mat",
                    "181025_pose_510.mat",
                    "181025_pose_511.mat",
                    "181025_pose_512.mat",
                    "181025_pose_513.mat",
                    "181025_pose_514.mat",
                    "181025_pose_515.mat"
                ]
                for i_fileResult in range(411, 511):
                    files_tobeCopied.append("181024_pose_%d.mat" % i_fileResult)
            try:
                for i_file in files_tobeCopied:
                    shutil.copy(pathSave + str_dir_final_last + i_file, path_save_direction + i_file)
            except:
                pass

        str_dir_level_12 = str_dir_level_1 + str_dir_final

        path_log = "F:/0/"

        pathSaveImage = path_save_direction + "%s_1_run_pose(large)/" % (str_timeDate_level_2)
        if PC in CameraSystem_L:
            utility.mkdir(pathSaveImage)

        pathFile_numCluster = path_save_direction + "num_cluster"

    # path read
    if 1:
        # path_read = path_dir + '/0 data_init/'
        path_dataInit = path_data
        # double
        path_TposeColor = path_dataInit + "color_blackWhite.mat"
        # path_TposeColor = path_save_direction + "color_180716_JiangYellow.mat"
        if 1:
            pathName_rl_inits = [
                "dataResult_180927_image1809_7_221/181007_5_shape.mat",
                "dataResult_181016_image1810_8/181022_1826_LegMul4/181016_5_shape.mat",
                "181016_5_shape.mat"
            ]
            pathfile_rl_init = path_data + pathName_rl_inits[-1]

        pathHalf_image = "0_image_1811_2_cali/K1_RT1"                       # warning1:new_data
        if PC == "Desktop":
            path_image_half = "X:/0_project_SoG/0_dataset/" + pathHalf_image
        elif PC in CameraSystem_L:
            path_image_half = "D:/" + pathHalf_image
        elif PC in CameraSystem_S:
            path_image_half = "D:/" + pathHalf_image
        else:
            assert False, "PC error"
        pathCameraParams = path_image_half + "/" + "data_detect/M_fl.mat"
        pathVideo_ = path_image_half + "_D1/"                                                # warning1:new_data

        path_imageCol_1 = pathVideo_ + "col/" + str(idImage[0]) + "-"
        path_imageCol_2 = pathVideo_ + "col/" + str(idImage[1]) + "-"

        # sil/
        name_sil = "sil_auto/"
        path_imageSil_1 = pathVideo_ + name_sil + str(idImage[0]) + "-"
        path_imageSil_2 = pathVideo_ + name_sil + str(idImage[1]) + "-"
        path_imageSil = pathVideo_ + name_sil

        pathFile_Tpose = path_data + "Tpose_br_11_1.mat"

# tmp
if 1:
    shape_delta = 99





