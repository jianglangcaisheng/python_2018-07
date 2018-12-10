
import tensorflow as tf
import numpy as np
import scipy.io as sio
import datetime
import math
import os

import z_main.function_0505.zz_func_getAllPreProduce as myPreProduce
import z_main.function_0505.EulerAndAxisAngle as EulerAndAxisAngle

def read_pose_init(path_pose_init):
    data = sio.loadmat(path_pose_init)
    # pose_init = data['pose_init']
    pose_init = data['axisAngle_3_24']

    axisAngle = EulerAndAxisAngle.pose2AxisAngle(input_pose=pose_init)
    pose_produceByAxisAngle = EulerAndAxisAngle.axisAngle2Pose(input_axisAngle=axisAngle)


    return [pose_init[0][0:69], np.transpose(pose_init[0][69:72])[:,np.newaxis]]

path_pose_init = "axisAngledetec_01_01_00.mat"
[pose_init, root_xyz] = read_pose_init(path_pose_init=path_pose_init)


