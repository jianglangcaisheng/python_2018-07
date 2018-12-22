
# _180914_can_run123

import tensorflow as tf
import numpy as np
import scipy.io as sio
import datetime
import math
import os
import cv2

import z_main.function_0505_2_crossPlatform.zz_func_getAllPreProduce as myPreProduce
import z_main.config as cf
import z_main.resource as rc
import z_main.function_0505_2_crossPlatform.module_Lab as module_Lab
import z_main.function_mini as function_mini
import z_main.function.image_produce as mImage_produce
from z_main.config_class import ModeConfig
import utility


def mkdir(path):
    import os

    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def update_Tpose_from_rl(body, rl):
    r = rl[0:1, 0:36]
    l = rl[0:1, 36:52]
    sigma = np.zeros([1, 63])
    sigma[0:1, 0:15] = r[0:1, 0:15]
    for i in range(11):
        sigma[0:1, 15+i] = r[0:1, 13-i]
    sigma[0:1, 26:41] = np.array([[r[0, 15], r[0, 16], r[0, 15], r[0, 17], r[0, 18], r[0, 17],
                                   r[0, 19], r[0, 20], r[0, 19], r[0, 21], r[0, 22], r[0, 21], r[0, 23], r[0, 24], r[0, 23]]])
    sigma[0:1, 41:63] = np.array([[r[0, 25], r[0, 25],
                                   r[0, 26], r[0, 26],
                                   r[0, 27], r[0, 27],
                                   r[0, 28], r[0, 28],
                                   r[0, 29], r[0, 29],
                                   r[0, 30], r[0, 30],
                                   r[0, 31], r[0, 31],
                                   r[0, 32], r[0, 32],
                                   r[0, 33], r[0, 33],
                                   r[0, 34], r[0, 34],
                                   r[0, 35], r[0, 35]]])

    # mu = np.array([
    #     [0, l(1) + l(2) + l(3) + l(6) + l(7), 0],        # 1
    #     [0, l(2) + l(3) + l(6) + l(7), 0],               # 2
    #     [0, l(3) + l(6) + l(7), 0],                      # 3
    #     [- l(4) - l(5) - l(8), l(6) + l(7), 0],          # 4
    #     [- 4 / 5 * l(4) - l(5) - l(8), l(6) + l(7), 0],  # 5
    #     [- 3 / 5 * l(4) - l(5) - l(8), l(6) + l(7), 0],  # 6
    #     [- 2 / 5 * l(4) - l(5) - l(8), l(6) + l(7), 0],  # 7
    #     [- 1 / 5 * l(4) - l(5) - l(8), l(6) + l(7), 0],  # 8
    #     [- l(5) - l(8), l(6) + l(7), 0],                 # 9
    #     [- 3 / 4 * l(5) - l(8), l(6) + l(7), 0],         # 10
    #     [- 2 / 4 * l(5) - l(8), l(6) + l(7), 0],         # 11
    #     [- 1 / 4 * l(5) - l(8), l(6) + l(7), 0],         # 12
    #     [- l(8), l(6) + l(7), 0],                        # 13
    #     [- 1 / 2 * l(8), l(6) + l(7), 0],                # 14
    #     [0, l(6) + l(7), 0],                             # 15
    #     [1 / 2 * l(8), l(6) + l(7), 0],                  # 16
    #     [l(8), l(6) + l(7), 0],                          # 17
    #     [1 / 4 * l(5) + l(8), l(6) + l(7), 0],           # 18
    #     [2 / 4 * l(5) + l(8), l(6) + l(7), 0],           # 19
    #     [3 / 4 * l(5) + l(8), l(6) + l(7), 0],           # 20
    #     [l(5) + l(8), l(6) + l(7), 0],                   # 21
    #     [1 / 5 * l(4) + l(5) + l(8), l(6) + l(7), 0],    # 22
    #     [2 / 5 * l(4) + l(5) + l(8), l(6) + l(7), 0],    # 23
    #     [3 / 5 * l(4) + l(5) + l(8), l(6) + l(7), 0],    # 24
    #     [4 / 5 * l(4) + l(5) + l(8), l(6) + l(7), 0],    # 25
    #     [l(4) + l(5) + l(8), l(6) + l(7), 0],            # 26
    #     [- l(9), 3 / 4 * l(6) + l(7), 0],                # 27
    #     [0, 3 / 4 * l(6) + l(7), 0],                     # 28
    #     [l(9), 3 / 4 * l(6) + l(7), 0],                  # 29
    #     [- l(10), 2 / 4 * l(6) + l(7), 0],               # 30
    #     [0, 2 / 4 * l(6) + l(7), 0],                     # 31
    #     [l(10), 2 / 4 * l(6) + l(7), 0],                 # 32
    #     [- l(11), 1 / 4 * l(6) + l(7), 0],               # 33
    #     [0, 1 / 4 * l(6) + l(7), 0],                     # 34
    #     [l(11), 1 / 4 * l(6) + l(7), 0],                 # 35
    #     [- l(12), l(7), 0],                              # 36
    #     [0, l(7), 0],                                    # 37
    #     [l(12), l(7), 0],                                # 38
    #     [- l(13), 0, 0],                                 # 39
    #     [0, 0, 0],                                       # 40
    #     [l(13), 0, 0],                                   # 41
    #     [- l(13), - 1 / 4 * l(14), 0],                   # 42
    #     [l(13), - 1 / 4 * l(14), 0],                     # 43
    #     [- l(13), - 2 / 4 * l(14), 0],                   # 44
    #     [l(13), - 2 / 4 * l(14), 0],                     # 45
    #     [- l(13), - 3 / 4 * l(14), 0],                   # 46
    #     [l(13), - 3 / 4 * l(14), 0],                     # 47
    #     [- l(13), - l(14), 0],                           # 48
    #     [l(13), - l(14), 0],                             # 49
    #     [- l(13), - 1 / 5 * l(15) - l(14), 0],           # 50
    #     [l(13), - 1 / 5 * l(15) - l(14), 0],             # 51
    #     [- l(13), - 2 / 5 * l(15) - l(14), 0],           # 52
    #     [l(13), - 2 / 5 * l(15) - l(14), 0],             # 53
    #     [- l(13), - 3 / 5 * l(15) - l(14), 0],           # 54
    #     [l(13), - 3 / 5 * l(15) - l(14), 0],             # 55
    #     [- l(13), - 4 / 5 * l(15) - l(14), 0],           # 56
    #     [l(13), - 4 / 5 * l(15) - l(14), 0],             # 57
    #     [- l(13), - l(15) - l(14), 0],                   # 58
    #     [l(13), - l(15) - l(14), 0],                     # 59
    #     [- l(13), - l(15) - l(14), 1 / 2 * l(16)],       # 60
    #     [l(13), - l(15) - l(14), 1 / 2 * l(16)],         # 61
    #     [- l(13), - l(15) - l(14), l(16)],               # 62
    #     [l(13), - l(15) - l(14), l(16)]])                # 63
    l = np.reshape(l, [16])
    mu = np.array([
        [0, l[0] + l[1] + l[2] + l[5] + l[6], 0],        # 1
        [0, l[1] + l[2] + l[5] + l[6], 0],               # 2
        [0, l[2] + l[5] + l[6], 0],                      # 3
        [- l[3] - l[4] - l[7], l[5] + l[6], 0],          # 4
        [- 4 / 5 * l[3] - l[4] - l[7], l[5] + l[6], 0],  # 5
        [- 3 / 5 * l[3] - l[4] - l[7], l[5] + l[6], 0],  # 6
        [- 2 / 5 * l[3] - l[4] - l[7], l[5] + l[6], 0],  # 7
        [- 1 / 5 * l[3] - l[4] - l[7], l[5] + l[6], 0],  # 8
        [- l[4] - l[7], l[5] + l[6], 0],                 # 9
        [- 3 / 4 * l[4] - l[7], l[5] + l[6], 0],         # 10
        [- 2 / 4 * l[4] - l[7], l[5] + l[6], 0],         # 11
        [- 1 / 4 * l[4] - l[7], l[5] + l[6], 0],         # 12
        [- l[7], l[5] + l[6], 0],                        # 13
        [- 1 / 2 * l[7], l[5] + l[6], 0],                # 14
        [0, l[5] + l[6], 0],                             # 15
        [1 / 2 * l[7], l[5] + l[6], 0],                  # 16
        [l[7], l[5] + l[6], 0],                          # 17
        [1 / 4 * l[4] + l[7], l[5] + l[6], 0],           # 18
        [2 / 4 * l[4] + l[7], l[5] + l[6], 0],           # 19
        [3 / 4 * l[4] + l[7], l[5] + l[6], 0],           # 20
        [l[4] + l[7], l[5] + l[6], 0],                   # 21
        [1 / 5 * l[3] + l[4] + l[7], l[5] + l[6], 0],    # 22
        [2 / 5 * l[3] + l[4] + l[7], l[5] + l[6], 0],    # 23
        [3 / 5 * l[3] + l[4] + l[7], l[5] + l[6], 0],    # 24
        [4 / 5 * l[3] + l[4] + l[7], l[5] + l[6], 0],    # 25
        [l[3] + l[4] + l[7], l[5] + l[6], 0],            # 26
        [- l[8], 3 / 4 * l[5] + l[6], 0],                # 27
        [0, 3 / 4 * l[5] + l[6], 0],                     # 28
        [l[8], 3 / 4 * l[5] + l[6], 0],                  # 29
        [- l[9], 2 / 4 * l[5] + l[6], 0],               # 30
        [0, 2 / 4 * l[5] + l[6], 0],                     # 31
        [l[9], 2 / 4 * l[5] + l[6], 0],                 # 32
        [- l[10], 1 / 4 * l[5] + l[6], 0],               # 33
        [0, 1 / 4 * l[5] + l[6], 0],                     # 34
        [l[10], 1 / 4 * l[5] + l[6], 0],                 # 35
        [- l[11], l[6], 0],                              # 36
        [0, l[6], 0],                                    # 37
        [l[11], l[6], 0],                                # 38
        [- l[12], 0, 0],                                 # 39
        [0, 0, 0],                                       # 40
        [l[12], 0, 0],                                   # 41
        [- l[12], - 1 / 4 * l[13], 0],                   # 42
        [l[12], - 1 / 4 * l[13], 0],                     # 43
        [- l[12], - 2 / 4 * l[13], 0],                   # 44
        [l[12], - 2 / 4 * l[13], 0],                     # 45
        [- l[12], - 3 / 4 * l[13], 0],                   # 46
        [l[12], - 3 / 4 * l[13], 0],                     # 47
        [- l[12], - l[13], 0],                           # 48
        [l[12], - l[13], 0],                             # 49
        [- l[12], - 1 / 5 * l[14] - l[13], 0],           # 50
        [l[12], - 1 / 5 * l[14] - l[13], 0],             # 51
        [- l[12], - 2 / 5 * l[14] - l[13], 0],           # 52
        [l[12], - 2 / 5 * l[14] - l[13], 0],             # 53
        [- l[12], - 3 / 5 * l[14] - l[13], 0],           # 54
        [l[12], - 3 / 5 * l[14] - l[13], 0],             # 55
        [- l[12], - 4 / 5 * l[14] - l[13], 0],           # 56
        [l[12], - 4 / 5 * l[14] - l[13], 0],             # 57
        [- l[12], - l[14] - l[13], 0],                   # 58
        [l[12], - l[14] - l[13], 0],                     # 59
        [- l[12], - l[14] - l[13], 1 / 2 * l[15]],       # 60
        [l[12], - l[14] - l[13], 1 / 2 * l[15]],         # 61
        [- l[12], - l[14] - l[13], l[15]],               # 62
        [l[12], - l[14] - l[13], l[15]]])                # 63
    dlt = np.concatenate([np.reshape(sigma, [63]), np.reshape(mu, [3*63])])
    body[0, 0][0, :] = dlt
    return body


def update_Tpose_from_rl_tensor(body, rl):
    r = rl[0:1, 0:36]       # 36
    l = rl[0:1, 36:52]      # 16
    # sigma = np.zeros([1, 63])
    sigma_1_0_15 = r[0:1, 0:15]
    for i in range(11):
        # sigma[0:1, 15+i] = r[0:1, 13-i]
        names["sigma_1_" + str(15 + i)] = r[0:1, 13-i:14-i]

    # sigma[0:1, 26:41] = np.array([[r[0, 15], r[0, 16], r[0, 15], r[0, 17], r[0, 18], r[0, 17],
    #                                r[0, 19], r[0, 20], r[0, 19], r[0, 21], r[0, 22], r[0, 21], r[0, 23], r[0, 24], r[0, 23]]])
    sigma_1_26_41 = tf.expand_dims(tf.stack([r[0, 15], r[0, 16], r[0, 15], r[0, 17], r[0, 18], r[0, 17],
                                   r[0, 19], r[0, 20], r[0, 19], r[0, 21], r[0, 22], r[0, 21], r[0, 23], r[0, 24], r[0, 23]], axis=0), axis=0)
    # sigma[0:1, 41:63] = np.array([[r[0, 25], r[0, 25],
    #                                r[0, 26], r[0, 26],
    #                                r[0, 27], r[0, 27],
    #                                r[0, 28], r[0, 28],
    #                                r[0, 29], r[0, 29],
    #                                r[0, 30], r[0, 30],
    #                                r[0, 31], r[0, 31],
    #                                r[0, 32], r[0, 32],
    #                                r[0, 33], r[0, 33],
    #                                r[0, 34], r[0, 34],
    #                                r[0, 35], r[0, 35]]])
    sigma_1_41_63 = tf.expand_dims(tf.stack([r[0, 25], r[0, 25],
                                   r[0, 26], r[0, 26],
                                   r[0, 27], r[0, 27],
                                   r[0, 28], r[0, 28],
                                   r[0, 29], r[0, 29],
                                   r[0, 30], r[0, 30],
                                   r[0, 31], r[0, 31],
                                   r[0, 32], r[0, 32],
                                   r[0, 33], r[0, 33],
                                   r[0, 34], r[0, 34],
                                   r[0, 35], r[0, 35]], axis=0), axis=0)
    sigma = tf.concat([sigma_1_0_15,
                       names["sigma_1_" + str(15 + 0)], names["sigma_1_" + str(15 + 1)],
                       names["sigma_1_" + str(15 + 2)], names["sigma_1_" + str(15 + 3)],
                       names["sigma_1_" + str(15 + 4)], names["sigma_1_" + str(15 + 5)],
                       names["sigma_1_" + str(15 + 6)], names["sigma_1_" + str(15 + 7)],
                       names["sigma_1_" + str(15 + 8)], names["sigma_1_" + str(15 + 9)],
                       names["sigma_1_" + str(15 + 10)], sigma_1_26_41, sigma_1_41_63], axis=1)
    # l = np.reshape(l, [16])
    l = tf.reshape(l, shape=[16])
    # mu = np.array([
    #     [0, l[0] + l[1] + l[2] + l[5] + l[6], 0],        # 1
    #     [0, l[1] + l[2] + l[5] + l[6], 0],               # 2
    #     [0, l[2] + l[5] + l[6], 0],                      # 3
    #     [- l[3] - l[4] - l[7], l[5] + l[6], 0],          # 4
    #     [- 4 / 5 * l[3] - l[4] - l[7], l[5] + l[6], 0],  # 5
    #     [- 3 / 5 * l[3] - l[4] - l[7], l[5] + l[6], 0],  # 6
    #     [- 2 / 5 * l[3] - l[4] - l[7], l[5] + l[6], 0],  # 7
    #     [- 1 / 5 * l[3] - l[4] - l[7], l[5] + l[6], 0],  # 8
    #     [- l[4] - l[7], l[5] + l[6], 0],                 # 9
    #     [- 3 / 4 * l[4] - l[7], l[5] + l[6], 0],         # 10
    #     [- 2 / 4 * l[4] - l[7], l[5] + l[6], 0],         # 11
    #     [- 1 / 4 * l[4] - l[7], l[5] + l[6], 0],         # 12
    #     [- l[7], l[5] + l[6], 0],                        # 13
    #     [- 1 / 2 * l[7], l[5] + l[6], 0],                # 14
    #     [0, l[5] + l[6], 0],                             # 15
    #     [1 / 2 * l[7], l[5] + l[6], 0],                  # 16
    #     [l[7], l[5] + l[6], 0],                          # 17
    #     [1 / 4 * l[4] + l[7], l[5] + l[6], 0],           # 18
    #     [2 / 4 * l[4] + l[7], l[5] + l[6], 0],           # 19
    #     [3 / 4 * l[4] + l[7], l[5] + l[6], 0],           # 20
    #     [l[4] + l[7], l[5] + l[6], 0],                   # 21
    #     [1 / 5 * l[3] + l[4] + l[7], l[5] + l[6], 0],    # 22
    #     [2 / 5 * l[3] + l[4] + l[7], l[5] + l[6], 0],    # 23
    #     [3 / 5 * l[3] + l[4] + l[7], l[5] + l[6], 0],    # 24
    #     [4 / 5 * l[3] + l[4] + l[7], l[5] + l[6], 0],    # 25
    #     [l[3] + l[4] + l[7], l[5] + l[6], 0],            # 26
    #     [- l[8], 3 / 4 * l[5] + l[6], 0],                # 27
    #     [0, 3 / 4 * l[5] + l[6], 0],                     # 28
    #     [l[8], 3 / 4 * l[5] + l[6], 0],                  # 29
    #     [- l[9], 2 / 4 * l[5] + l[6], 0],               # 30
    #     [0, 2 / 4 * l[5] + l[6], 0],                     # 31
    #     [l[9], 2 / 4 * l[5] + l[6], 0],                 # 32
    #     [- l[10], 1 / 4 * l[5] + l[6], 0],               # 33
    #     [0, 1 / 4 * l[5] + l[6], 0],                     # 34
    #     [l[10], 1 / 4 * l[5] + l[6], 0],                 # 35
    #     [- l[11], l[6], 0],                              # 36
    #     [0, l[6], 0],                                    # 37
    #     [l[11], l[6], 0],                                # 38
    #     [- l[12], 0, 0],                                 # 39
    #     [0, 0, 0],                                       # 40
    #     [l[12], 0, 0],                                   # 41
    #     [- l[12], - 1 / 4 * l[13], 0],                   # 42
    #     [l[12], - 1 / 4 * l[13], 0],                     # 43
    #     [- l[12], - 2 / 4 * l[13], 0],                   # 44
    #     [l[12], - 2 / 4 * l[13], 0],                     # 45
    #     [- l[12], - 3 / 4 * l[13], 0],                   # 46
    #     [l[12], - 3 / 4 * l[13], 0],                     # 47
    #     [- l[12], - l[13], 0],                           # 48
    #     [l[12], - l[13], 0],                             # 49
    #     [- l[12], - 1 / 5 * l[14] - l[13], 0],           # 50
    #     [l[12], - 1 / 5 * l[14] - l[13], 0],             # 51
    #     [- l[12], - 2 / 5 * l[14] - l[13], 0],           # 52
    #     [l[12], - 2 / 5 * l[14] - l[13], 0],             # 53
    #     [- l[12], - 3 / 5 * l[14] - l[13], 0],           # 54
    #     [l[12], - 3 / 5 * l[14] - l[13], 0],             # 55
    #     [- l[12], - 4 / 5 * l[14] - l[13], 0],           # 56
    #     [l[12], - 4 / 5 * l[14] - l[13], 0],             # 57
    #     [- l[12], - l[14] - l[13], 0],                   # 58
    #     [l[12], - l[14] - l[13], 0],                     # 59
    #     [- l[12], - l[14] - l[13], 1 / 2 * l[15]],       # 60
    #     [l[12], - l[14] - l[13], 1 / 2 * l[15]],         # 61
    #     [- l[12], - l[14] - l[13], l[15]],               # 62
    #     [l[12], - l[14] - l[13], l[15]]])                # 63
    mu = tf.stack([
        tf.stack([0, l[0] + l[1] + l[2] + l[5] + l[6], 0], axis=0),  # 1
        tf.stack([0, l[1] + l[2] + l[5] + l[6], 0], axis=0),  # 2
        tf.stack([0, l[2] + l[5] + l[6], 0], axis=0),  # 3
        tf.stack([- l[3] - l[4] - l[7], l[5] + l[6], 0], axis=0),  # 4
        tf.stack([- 4 / 5 * l[3] - l[4] - l[7], l[5] + l[6], 0], axis=0),  # 5
        tf.stack([- 3 / 5 * l[3] - l[4] - l[7], l[5] + l[6], 0], axis=0),  # 6
        tf.stack([- 2 / 5 * l[3] - l[4] - l[7], l[5] + l[6], 0], axis=0),  # 7
        tf.stack([- 1 / 5 * l[3] - l[4] - l[7], l[5] + l[6], 0], axis=0),  # 8
        tf.stack([- l[4] - l[7], l[5] + l[6], 0], axis=0),  # 9
        tf.stack([- 3 / 4 * l[4] - l[7], l[5] + l[6], 0], axis=0),  # 10
        tf.stack([- 2 / 4 * l[4] - l[7], l[5] + l[6], 0], axis=0),  # 11
        tf.stack([- 1 / 4 * l[4] - l[7], l[5] + l[6], 0], axis=0),  # 12
        tf.stack([- l[7], l[5] + l[6], 0], axis=0),  # 13
        tf.stack([- 1 / 2 * l[7], l[5] + l[6], 0], axis=0),  # 14
        tf.stack([0, l[5] + l[6], 0], axis=0),  # 15
        tf.stack([1 / 2 * l[7], l[5] + l[6], 0], axis=0),  # 16
        tf.stack([l[7], l[5] + l[6], 0], axis=0),  # 17
        tf.stack([1 / 4 * l[4] + l[7], l[5] + l[6], 0], axis=0),  # 18
        tf.stack([2 / 4 * l[4] + l[7], l[5] + l[6], 0], axis=0),  # 19
        tf.stack([3 / 4 * l[4] + l[7], l[5] + l[6], 0], axis=0),  # 20
        tf.stack([l[4] + l[7], l[5] + l[6], 0], axis=0),  # 21
        tf.stack([1 / 5 * l[3] + l[4] + l[7], l[5] + l[6], 0], axis=0),  # 22
        tf.stack([2 / 5 * l[3] + l[4] + l[7], l[5] + l[6], 0], axis=0),  # 23
        tf.stack([3 / 5 * l[3] + l[4] + l[7], l[5] + l[6], 0], axis=0),  # 24
        tf.stack([4 / 5 * l[3] + l[4] + l[7], l[5] + l[6], 0], axis=0),  # 25
        tf.stack([l[3] + l[4] + l[7], l[5] + l[6], 0], axis=0),  # 26
        tf.stack([- l[8], 3 / 4 * l[5] + l[6], 0], axis=0),  # 27
        tf.stack([0, 3 / 4 * l[5] + l[6], 0], axis=0),  # 28
        tf.stack([l[8], 3 / 4 * l[5] + l[6], 0], axis=0),  # 29
        tf.stack([- l[9], 2 / 4 * l[5] + l[6], 0], axis=0),  # 30
        tf.stack([0, 2 / 4 * l[5] + l[6], 0], axis=0),  # 31
        tf.stack([l[9], 2 / 4 * l[5] + l[6], 0], axis=0),  # 32
        tf.stack([- l[10], 1 / 4 * l[5] + l[6], 0], axis=0),  # 33
        tf.stack([0, 1 / 4 * l[5] + l[6], 0], axis=0),  # 34
        tf.stack([l[10], 1 / 4 * l[5] + l[6], 0], axis=0),  # 35
        tf.stack([- l[11], l[6], 0], axis=0),  # 36
        tf.stack([0, l[6], 0], axis=0),  # 37
        tf.stack([l[11], l[6], 0], axis=0),  # 38
        tf.stack([- l[12], 0, 0], axis=0),  # 39
        tf.stack([0., 0., 0.], axis=0),  # 40
        tf.stack([l[12], 0, 0], axis=0),  # 41
        tf.stack([- l[12], - 1 / 4 * l[13], 0], axis=0),  # 42
        tf.stack([l[12], - 1 / 4 * l[13], 0], axis=0),  # 43
        tf.stack([- l[12], - 2 / 4 * l[13], 0], axis=0),  # 44
        tf.stack([l[12], - 2 / 4 * l[13], 0], axis=0),  # 45
        tf.stack([- l[12], - 3 / 4 * l[13], 0], axis=0),  # 46
        tf.stack([l[12], - 3 / 4 * l[13], 0], axis=0),  # 47
        tf.stack([- l[12], - l[13], 0], axis=0),  # 48
        tf.stack([l[12], - l[13], 0], axis=0),  # 49
        tf.stack([- l[12], - 1 / 5 * l[14] - l[13], 0], axis=0),  # 50
        tf.stack([l[12], - 1 / 5 * l[14] - l[13], 0], axis=0),  # 51
        tf.stack([- l[12], - 2 / 5 * l[14] - l[13], 0], axis=0),  # 52
        tf.stack([l[12], - 2 / 5 * l[14] - l[13], 0], axis=0),  # 53
        tf.stack([- l[12], - 3 / 5 * l[14] - l[13], 0], axis=0),  # 54
        tf.stack([l[12], - 3 / 5 * l[14] - l[13], 0], axis=0),  # 55
        tf.stack([- l[12], - 4 / 5 * l[14] - l[13], 0], axis=0),  # 56
        tf.stack([l[12], - 4 / 5 * l[14] - l[13], 0], axis=0),  # 57
        tf.stack([- l[12], - l[14] - l[13], 0], axis=0),  # 58
        tf.stack([l[12], - l[14] - l[13], 0], axis=0),  # 59
        tf.stack([- l[12], - l[14] - l[13], 1 / 2 * l[15]], axis=0),  # 60
        tf.stack([l[12], - l[14] - l[13], 1 / 2 * l[15]], axis=0),  # 61
        tf.stack([- l[12], - l[14] - l[13], l[15]], axis=0),  # 62
        tf.stack([l[12], - l[14] - l[13], l[15]], axis=0)], axis=0)  # 63
    # dlt = np.concatenate([np.reshape(sigma, [63]), np.reshape(mu, [3*63])])
    dlt = tf.concat([tf.reshape(sigma, shape=[1, 63]), tf.reshape(mu, shape=[1, 3*63])], axis=1)
    # body[0, 0][0, :] = dlt
    return dlt


def draw_body(mode, mode_image, coordinate_2D, radius_2D, color=None, windowName=None, note=None, imageClustered=None,
              pInModeCS=None, mCS_iShape=0, mImO_imagePath=None):

    # Lab, RGB
    for mode_colorSpace in ["RGB"]:

        for i_num_view in range(cf.num_view):

            # draw
            if mode_image == "cluster":
                imageCopy = imageClustered[i_num_view].copy()
            elif mode_image == "origin":
                [pathVideo_, iImage] = mImO_imagePath
                pathFile_image = pathVideo_ + ("%d_bmp/%d-%d.bmp" % (i_num_view, iImage, i_num_view))
                imageCopy = cv2.imread(pathFile_image)

            if mode_colorSpace == "Lab":
                imageCopy =module_Lab.BGR2RGB(imageCopy)
                imageCopy = module_Lab.RGB2LAB(imageCopy)
                imageCopy[:, :, 0] = function_mini.compress_L_in_Lab(imageCopy[:, :, 0])
                imageCopy = module_Lab.LAB2RGB(imageCopy)
                imageCopy = module_Lab.RGB2BGR(imageCopy)

            for i_bodyNumBall in range(cf.bodyNumBall):

                # circle
                if 1:
                    m = int(coordinate_2D[i_num_view][i_bodyNumBall][0])
                    n = int(coordinate_2D[i_num_view][i_bodyNumBall][1])
                    r = int(radius_2D[i_num_view][i_bodyNumBall])
                    if (r < 0):
                        print(windowName + ". m: " + str(m) + ", n: " + str(n) + ", r: " + str(r))
                        assert False, "r<0"

                    colorBall = np.array([color[i_bodyNumBall][2], color[i_bodyNumBall][1], color[i_bodyNumBall][0]]) * 255.

                    if mode_colorSpace == "Lab":
                        colorBall = np.reshape(colorBall, [1, 1, 3])
                        colorBall = module_Lab.BGR2RGB(colorBall)
                        colorBall = module_Lab.RGB2LAB(colorBall)
                        colorBall[:, :, 0] = function_mini.compress_L_in_Lab(colorBall[:, :, 0])
                        colorBall = module_Lab.LAB2RGB(colorBall)
                        colorBall = module_Lab.RGB2BGR(colorBall)
                        colorBall = np.reshape(colorBall, [3])


                    cv2.circle(imageCopy, (m, n), r, np.float64(colorBall), 8)
                    if i_bodyNumBall + 1 in [1, 5, 9, 13, 15, 17, 21, 25, 28, 31, 34, 37, 39, 40, 41, 48, 49, 58, 59]:
                        cv2.circle(imageCopy, (m, n), 8, (0, 0, 255), -1)

                # space of getting color
                if mode == "circle_and_square" and 0:

                    [row_down, row_up, col_down, col_up] = pInModeCS[i_bodyNumBall][mCS_iShape][i_num_view]

                    cv2.rectangle(imageCopy, (row_down, col_down), (row_up, col_up), color=(255, 0, 0), thickness=4)

            # save
            pathSave_image = cf.pathSaveImage + note + "_c" + mode_colorSpace + "_v" + str(i_num_view) + ".png"
            result = cv2.imwrite(pathSave_image, imageCopy, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            if result == False:
                print("pathSave_image: %s" % pathSave_image)
                print("pathFile_image: %s" % pathFile_image)
                assert False, "Failure imwrite"

    # imshow in running
    if 0:
        cv2.namedWindow(windowName, 0)
        cv2.resizeWindow(windowName, 1024, 768)
        cv2.imshow(windowName, imageCopy)
        cv2.waitKey(0)

    return


def draw_body2(images, data_result, cf_for_drawBody):
    coordinate_2D = data_result[0]
    radius_2D = data_result[1]
    color = data_result[2]
    mode = cf_for_drawBody.mode
    mode_image = cf_for_drawBody.mode_image
    note = cf_for_drawBody.note
    windowName = cf_for_drawBody.windowName
    pInModeCS = cf_for_drawBody.pInModeCS

    # Lab, RGB
    for mode_colorSpace in ["RGB"]:
        for i_num_view in range(cf.num_view):

            # draw
            if mode_image == "cluster":
                imageCopy = images[i_num_view].copy()
            elif mode_image == "origin":
                while True:
                    i_frame_open = cf_for_drawBody.i_frame
                    pathFile_image = cf.pathVideo_ + ("%d_bmp/%d-%d.bmp" % (i_num_view, i_frame_open, i_num_view))
                    try:
                        imageCopy = cv2.imread(pathFile_image)
                        break
                    except:
                        utility.print_red("Error pathFile_image: %s" % pathFile_image)
                        i_frame_open = i_frame_open - 1
                        if i_frame_open < 0:
                            assert False, "i_frame_open < 0"
            else:
                assert False, "mode_image doesn't has: %s" % mode_image

            if mode_colorSpace == "Lab":
                imageCopy =module_Lab.BGR2RGB(imageCopy)
                imageCopy = module_Lab.RGB2LAB(imageCopy)
                imageCopy[:, :, 0] = function_mini.compress_L_in_Lab(imageCopy[:, :, 0])
                imageCopy = module_Lab.LAB2RGB(imageCopy)
                imageCopy = module_Lab.RGB2BGR(imageCopy)

            for i_bodyNumBall in range(cf.bodyNumBall):

                # circle
                if 1:
                    m = int(coordinate_2D[i_num_view][i_bodyNumBall][0])
                    n = int(coordinate_2D[i_num_view][i_bodyNumBall][1])
                    r = int(radius_2D[i_num_view][i_bodyNumBall])
                    if (r < 0):
                        print(windowName + ". m: " + str(m) + ", n: " + str(n) + ", r: " + str(r))
                        assert False, "r<0"

                    colorBall = np.array([color[i_bodyNumBall][2], color[i_bodyNumBall][1], color[i_bodyNumBall][0]]) * 255.

                    if mode_colorSpace == "Lab":
                        colorBall = np.reshape(colorBall, [1, 1, 3])
                        colorBall = module_Lab.BGR2RGB(colorBall)
                        colorBall = module_Lab.RGB2LAB(colorBall)
                        colorBall[:, :, 0] = function_mini.compress_L_in_Lab(colorBall[:, :, 0])
                        colorBall = module_Lab.LAB2RGB(colorBall)
                        colorBall = module_Lab.RGB2BGR(colorBall)
                        colorBall = np.reshape(colorBall, [3])


                    cv2.circle(imageCopy, (m, n), r, np.float64(colorBall), 8)
                    if i_bodyNumBall + 1 in [1, 5, 9, 13, 15, 17, 21, 25, 28, 31, 34, 37, 39, 40, 41, 48, 49, 58, 59]:
                        cv2.circle(imageCopy, (m, n), 8, (0, 0, 255), -1)

                # space of getting color
                if mode == "circle_and_square" and 0:

                    [row_down, row_up, col_down, col_up] = pInModeCS[i_bodyNumBall][mCS_iShape][i_num_view]

                    cv2.rectangle(imageCopy, (row_down, col_down), (row_up, col_up), color=(255, 0, 0), thickness=4)

            if 1:
                i_frame_load = cf_for_drawBody.i_frame
                while True:
                    try:
                        pathfile_YOLO_label = cf.path.YOLO_label + "%d-%d.mat" % (i_frame_load, i_num_view)
                        data = sio.loadmat(pathfile_YOLO_label)
                        break
                    except:
                        utility.print_red("Not exists: %s" % pathfile_YOLO_label)
                        i_frame_load = i_frame_load - 1

                bounding_box = data['bounding_box']
                cv2.rectangle(imageCopy, (bounding_box[0, 0], bounding_box[0, 1]), (bounding_box[0, 2], bounding_box[0, 3]), (0, 0, 255), 8)

                from function.image_produce import broad_boundingBox
                bounding_box, broad = broad_boundingBox(bounding_box, cf.params.broad)
                cv2.rectangle(imageCopy, (bounding_box[0, 0], bounding_box[0, 1]), (bounding_box[0, 2], bounding_box[0, 3]), (0, 0, 255), 4)

                font = cv2.FONT_HERSHEY_SIMPLEX
                imgzi = cv2.putText(imageCopy, str(broad), (50, 300), font, 1.2, (0, 0, 255), 2)

            # save
            if 0:
                pathSave_image = cf.pathSaveImage + note + "_c" + mode_colorSpace + "_v" + str(i_num_view) + ".png"
                result = cv2.imwrite(pathSave_image, imageCopy, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            else:
                pathSave_image = cf.pathSaveImage + note + "_c" + mode_colorSpace + "_v" + str(i_num_view) + ".jpg"
                result = cv2.imwrite(pathSave_image, imageCopy, [int(cv2.IMWRITE_JPEG_OPTIMIZE), 0])

            if result == False:
                print("pathSave_image: %s" % pathSave_image)
                print("pathFile_image: %s" % pathFile_image)
                utility.print_red("Failure imwrite: %s" % pathSave_image )
                # assert False, "Failure imwrite"

    # imshow in running
    if 0:
        cv2.namedWindow(windowName, 0)
        cv2.resizeWindow(windowName, 1024, 768)
        cv2.imshow(windowName, imageCopy)
        cv2.waitKey(0)

    return

# ...........................................................................................................................

def read_rl(path_rl):
    data = sio.loadmat(path_rl)
    rl = data['rl']
    return rl


def read_Tpose(path_Tpose, rl_tensor):
    data = sio.loadmat(path_Tpose)
    body = data['body']

    # rl = read_rl("F:/0 SOG/2018-03-30/matlab/code/2018-05-14 shape/brrl0516_init.mat")

    # body = update_Tpose_from_rl(body, rl)
    coordinate_Tpose_tensor = update_Tpose_from_rl_tensor(body, rl_tensor)          # 63*3

    # coordinate_Tpose = body_new[0][0]
    # coordinate_Tpose = body[0][0]
    return coordinate_Tpose_tensor


def read_bodyColor(path_bodyColor):
    data = sio.loadmat(path_bodyColor)
    bodyColor = data['color']
    return bodyColor


def read_pose_init(path_pose_init):
    data = sio.loadmat(path_pose_init)
    # pose_init = data['pose_init']
    pose_init = data['pose']
    return [pose_init[0][0:69], np.transpose(pose_init[0][69:72])[:,np.newaxis]]


def read_axisAngle_init(path_pose_init):
    data = sio.loadmat(path_pose_init)
    # pose_init = data['pose_init']
    pose_init = data['axisAngle_3_24']
    return pose_init


def read_image(path_image):
    data = sio.loadmat(path_image)

    B_mu = data['B_mu_new']
    B_sigma = data['B_sigma_new']
    D = data['D_new']
    EII = data['EII_new']
    size_of_max = data['size_of_cluster_ave']

    return [B_mu, B_sigma, D, EII, size_of_max]


def read_image_npz(path_image):
    npzfile = np.load(path_image)

    B_mu = npzfile['B_mu_new_fs']
    B_sigma = npzfile['B_sigma_new_fs']
    D = npzfile['D_new_fs']
    EII = npzfile['EII_new_fs']

    return [B_mu, B_sigma, D, EII]


def read_M_matrix(path_M_matrix):
    data = sio.loadmat(path_M_matrix)
    M_matrix = data['M']
    fl = data['fl']
    return [M_matrix, fl]


def SaveAxisAngle(path_save, axisAngle_3_24=None, filename_save="dlt.mat"):
    sio.savemat(path_save + '/' + filename_save,
                    {'axisAngle_3_24': axisAngle_3_24})


def SaveRL(path_save, rl=None, filename_save = "dlt.mat"):
    sio.savemat(path_save + '/' + filename_save,
                    {'rl': rl})


def run(ifPose = True, filename_pose="", filename_rl="", filename_save="", note_out="", ifAllVideo = False, i_image=1,
        path_TposeColor="", i_times=0, i_poseth=0, frame_start=0, num_frames=1, mode="", max_of_cluster=None):

    """mode: pose_in_all_video, expect color, sil_for_deep"""

    i_image = i_poseth

    # ifPose = False
    ifShape = not ifPose

    # todo: small
    if i_image == 0:
        idImage = cf.idImage[0]
    elif i_image == 1:
        idImage = cf.idImage[1]

    if ifAllVideo == True:
        path_video = cf.pathVideo_
    else:
        path_video = cf.path_imageSil + str(idImage) + "-"

    path_rl = filename_rl
    path_save = cf.pathSave
    video_choose = ''


    # 输出注释
    note_simple = "_3e8_2k" + note_out


    path_pose_init = filename_pose

    # canshu............................................................................................................
    if ifAllVideo == False:
        train_times = cf.trainTimes_poseInShape_array[i_times]
    elif ifAllVideo == True:
        train_times = cf.trainTimes_poseInPose
    if mode == "expect color":
        train_times = 1

    if ifPose == True:
        train_step = cf.trainStepPose
    elif ifShape ==True:
        train_step = 0.00000001 * 30 * 10 * 10
    else:
        train_step = 0
        print("No Trainable Variables")
        exit(1)


    # switch
    switch_writer = False
    if_save_time = False
    if switch_writer == False:
        if_save_time = False

    if_save_result = False  # no influence on time

    # preProduce................................................................................................................
    note = "_" + video_choose + note_simple

    # filename_image = '0 data_image/imageNew_' + video_choose + '.mat'

    begin = datetime.datetime.now()


    if if_save_result == True:
        mkdir(path_save)

    path_Tpose = cf.pathFile_Tpose

    path_M_matrix = cf.pathCameraParams

    mkdir(path=cf.path_log + cf.str_dir_level_12)
    log_path = cf.path_log + cf.str_dir_level_12 + begin.strftime('%m%d') + begin.strftime('_%H%M') + note


    TYPE = tf.float32

    # data
    joint = np.array([15, 5, 9, 13, 15, 15, 17, 21, 25, 28,
                       31, 34, 37, 40, 40, 40, 39, 41, 48, 49,
                       58, 59, 40])

    # main..................................................................................................................

    M_matrix, fl = read_M_matrix(path_M_matrix)

    with tf.name_scope('body'):
        rl = read_rl(path_rl=path_rl)
        rl_tensor = tf.Variable(rl, dtype=TYPE, trainable=ifShape, name="rl")
        Tpose = read_Tpose(path_Tpose=path_Tpose, rl_tensor=rl_tensor)
        radius_tensor = Tpose[0:1, 0:63]                                        # 1*63
        # coordinate_Tpose = Tpose[0][63:252][np.newaxis, :].reshape(3, 63, order='F')     # 3*63
        coordinate_Tpose_tensor = tf.reshape(Tpose[0:1, 63:252], shape=[63, 3])

        for i_pose69 in range(0, 69):
            i_pose = i_pose69 // 3      # 第几个关节
            j_pose = i_pose69 % 3       # 第几个
            temp2 = joint[i_pose]       # 旋转节点坐标
            # todo 前后优化
            names["T_vector_old_co_" + str(j_pose) + "_" + str(i_pose) + "_tensor"] = coordinate_Tpose_tensor[joint[i_pose] - 1, 0:3]
        for j_pose in range(3):
            names["T_vector_old_co_" + str(j_pose) + "_tensor"] = tf.stack([names["T_vector_old_co_" + str(j_pose) + "_" + str(0) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(1) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(2) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(3) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(4) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(5) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(6) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(7) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(8) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(9) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(10) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(11) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(12) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(13) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(14) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(15) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(16) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(17) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(18) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(19) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(20) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(21) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(22) + "_tensor"]], axis=0)
        T_vector_old_co_tensor = tf.stack([names["T_vector_old_co_" + str(0) + "_tensor"],
                                           names["T_vector_old_co_" + str(1) + "_tensor"],
                                           names["T_vector_old_co_" + str(2) + "_tensor"]], axis=0)

    with tf.name_scope('pose'):
        axisAngle_3_24_init = read_axisAngle_init(path_pose_init=path_pose_init)


        if 1:
            pathFile_step = cf.path_data + "0_data_tmp/step_7_rootAngle16.mat"

            data_step_rate = sio.loadmat(pathFile_step)
            data_step_rate = data_step_rate['data_step_rate']
            if 0:
                print("\n")
                print(data_step_rate)

            data_step_rate = np.maximum(data_step_rate, 1e-8)
            data_step_rate = np.divide(1, data_step_rate)
            data_step_rate = np.sqrt(data_step_rate)

        # speed
        if 1:
            axisAngle_3_20_init = np.concatenate([
                axisAngle_3_24_init[:, 0:10],
                axisAngle_3_24_init[:, 14:24]], axis=1)

            axisAngle_3_1_9_tensor = tf.multiply(tf.Variable(initial_value=axisAngle_3_20_init[:, 0:9] * data_step_rate[:, 0:9],
                                                             name='axisAngle3', dtype=TYPE, trainable=ifPose), 1 / data_step_rate[:, 0:9])
            axisAngle_3_10_10_tensor = tf.multiply(tf.Variable(initial_value=axisAngle_3_20_init[:, 9:10] * data_step_rate[:, 9:10],
                                                               name='axisAngle2', dtype=TYPE, trainable=ifPose), 1 / data_step_rate[:, 9:10])
            axisAngle_3_11_19_tensor = tf.multiply(tf.Variable(initial_value=axisAngle_3_20_init[:, 10:19] * data_step_rate[:, 14:23],
                                                               name='axisAngle2', dtype=TYPE, trainable=ifPose), 1 / data_step_rate[:, 14:23])

            axisAngle_3_19_tensor = tf.concat(
                [axisAngle_3_1_9_tensor, axisAngle_3_10_10_tensor, axisAngle_3_11_19_tensor], axis=1)
            axisAngle_3_23_tensor = tf.concat([axisAngle_3_19_tensor[:, 0:10],
                                               axisAngle_3_19_tensor[:, 9:10],
                                               axisAngle_3_19_tensor[:, 9:10],
                                               axisAngle_3_19_tensor[:, 9:10],
                                               axisAngle_3_19_tensor[:, 9:10],
                                               axisAngle_3_19_tensor[:, 10:19]], axis=1)

            root_variable = tf.multiply(tf.Variable(initial_value=axisAngle_3_24_init[:, 23:24] * data_step_rate[:, 23:24],
                                                    dtype=TYPE, name='root/173.', trainable=ifPose), 1 / data_step_rate[:, 23:24], name='root_31')

        root_tensor = tf.concat([root_variable, tf.zeros(shape=[1, 1])], axis=0, name='root_41')

        #
        theta = tf.sqrt(tf.multiply(axisAngle_3_23_tensor[0:1, 0:23], axisAngle_3_23_tensor[0:1, 0:23]) +
                        tf.multiply(axisAngle_3_23_tensor[1:2, 0:23], axisAngle_3_23_tensor[1:2, 0:23]) +
                        tf.multiply(axisAngle_3_23_tensor[2:3, 0:23], axisAngle_3_23_tensor[2:3, 0:23]) + 1e-9)

        axisAngle_4_23_axis = axisAngle_3_23_tensor / (theta)

    with tf.name_scope('rota_ma'):
        # todo test
        zeros_23 = tf.zeros(shape=[23])
        ones_23 = tf.ones(shape=[23])

        wx_23_1_3_first =  tf.stack([                   zeros_23, -axisAngle_4_23_axis[3 - 1],  axisAngle_4_23_axis[2 - 1]], axis=1, name='wx_23_1_3_first')
        wx_23_1_3_second = tf.stack([ axisAngle_4_23_axis[3 - 1],                    zeros_23, -axisAngle_4_23_axis[1 - 1]], axis=1, name='wx_23_1_3_second')
        wx_23_1_3_third =  tf.stack([-axisAngle_4_23_axis[2 - 1],  axisAngle_4_23_axis[1 - 1],                   zeros_23,], axis=1, name='wx_23_1_3_third')
        # wx_23_3_3 = tf.stack([wx_23_1_3_first, wx_23_1_3_second, wx_23_1_3_third], axis=1, name='wx_23_3_3')
        wx_23_3_3 = tf.stack([wx_23_1_3_first, wx_23_1_3_second, wx_23_1_3_third], axis=1, name='wx_23_3_3')

        diag = tf.tile(tf.expand_dims(tf.diag([1., 1., 1.]), axis=0), multiples=[23, 1, 1], name='diag')

        R_23_3_3 = diag + wx_23_3_3 * tf.tile(tf.expand_dims(tf.expand_dims(tf.sin(theta[0]), axis=1), axis=1), [1, 3, 3]) + \
                   tf.matmul(wx_23_3_3, wx_23_3_3) * tf.tile(tf.expand_dims(tf.expand_dims(1 - tf.cos(theta[0]), axis=1), axis=1), [1, 3, 3])

        theta_x_tensor = tf.atan2(R_23_3_3[:, 3 - 1, 2 - 1],
                                  R_23_3_3[:, 3 - 1, 3 - 1])
        theta_y_tensor = tf.atan2(R_23_3_3[:, 3 - 1, 1 - 1],
                                  tf.sqrt(R_23_3_3[:, 3 - 1, 2 - 1] *
                                          R_23_3_3[:, 3 - 1, 2 - 1] +
                                          R_23_3_3[:, 3 - 1, 3 - 1] *
                                          R_23_3_3[:, 3 - 1, 3 - 1]))
        theta_z_tensor = tf.atan2(R_23_3_3[:, 2 - 1, 1 - 1],
                                  R_23_3_3[:, 1 - 1, 1 - 1])

        euler_tensor = tf.stack([theta_x_tensor[0:22],
                                 theta_y_tensor[0:22],
                                 theta_z_tensor[0:22]], axis=1)

        #
        # rotation_matrix_0_0 = tf.stack([ones_23, zeros_23, zeros_23], axis=1, name='rota_0_0')
        # rotation_matrix_0_1 = tf.stack(
        #     [zeros_23, pose_3_23_tensor_cos_0, -pose_3_23_tensor_sin_0], axis=1, name='rota_0_1')
        # rotation_matrix_0_2 = tf.stack(
        #     [zeros_23, pose_3_23_tensor_sin_0, pose_3_23_tensor_cos_0], axis=1, name='rota_0_2')
        # rotation_matrix_0_sum = tf.stack([rotation_matrix_0_0, rotation_matrix_0_1, rotation_matrix_0_2], axis=1, name='rota_0_sum')
        #
        # rotation_matrix_1_0 = tf.stack(
        #     [pose_3_23_tensor_cos_1, zeros_23, pose_3_23_tensor_sin_1], axis=1, name='rota_1_0')
        # rotation_matrix_1_1 = tf.stack([zeros_23, ones_23, zeros_23], axis=1, name='rota_1_1')
        # rotation_matrix_1_2 = tf.stack(
        #     [-pose_3_23_tensor_sin_1, zeros_23, pose_3_23_tensor_cos_1], axis=1, name='rota_1_2')
        # rotation_matrix_1_sum = tf.stack([rotation_matrix_1_0, rotation_matrix_1_1, rotation_matrix_1_2], axis=1, name='rota_1_sum')
        #
        # rotation_matrix_2_0 = tf.stack(
        #     [pose_3_23_tensor_cos_2, -pose_3_23_tensor_sin_2, zeros_23], axis=1, name='rota_2_0')
        # rotation_matrix_2_1 = tf.stack(
        #     [pose_3_23_tensor_sin_2, pose_3_23_tensor_cos_2, zeros_23], axis=1, name='rota_2_1')
        # rotation_matrix_2_2 = tf.stack([zeros_23, zeros_23, ones_23], axis=1, name='rota_2_2')
        # rotation_matrix_2_sum = tf.stack([rotation_matrix_2_0, rotation_matrix_2_1, rotation_matrix_2_2], axis=1, name='rota_2_sum')
        #
        # # 3 * 23 * 3 * 3
        # rotation_matrix_sum_sum = tf.stack([rotation_matrix_0_sum, rotation_matrix_1_sum, rotation_matrix_2_sum], axis=0, name='rota_s_sum')

    with tf.name_scope('T_vec'):
            coordinate_joint_tensor = tf.expand_dims(T_vector_old_co_tensor[0, :, :], axis=2, name='co_joint')
            diag_23_3_3 = tf.tile(tf.expand_dims(tf.diag([1., 1., 1.]), axis=0), multiples=[23, 1, 1], name='diag')
            temp = (diag_23_3_3 - R_23_3_3)
            T_vector_tensor = tf.matmul(temp, coordinate_joint_tensor, name='T_vector')


    with tf.name_scope('tran_ma'):
        zero3one1 = tf.concat([tf.zeros(shape=[23, 1, 3]), tf.ones(shape=[23, 1, 1])], axis=2)
        transition_matrix_tensor = tf.concat([tf.concat([R_23_3_3, T_vector_tensor], axis=2), zero3one1], axis=1, name='transi')
        transition_matrix_3pose_tensor = transition_matrix_tensor

        transition_matrix_3pose_tensor_0 = transition_matrix_3pose_tensor[0]
        transition_matrix_3pose_tensor_1 = transition_matrix_3pose_tensor[1]
        transition_matrix_3pose_tensor_2 = transition_matrix_3pose_tensor[2]
        transition_matrix_3pose_tensor_3 = transition_matrix_3pose_tensor[3]
        transition_matrix_3pose_tensor_4 = transition_matrix_3pose_tensor[4]
        transition_matrix_3pose_tensor_5 = transition_matrix_3pose_tensor[5]
        transition_matrix_3pose_tensor_6 = transition_matrix_3pose_tensor[6]
        transition_matrix_3pose_tensor_7 = transition_matrix_3pose_tensor[7]
        transition_matrix_3pose_tensor_8 = transition_matrix_3pose_tensor[8]
        transition_matrix_3pose_tensor_9 = transition_matrix_3pose_tensor[9]
        transition_matrix_3pose_tensor_10 = transition_matrix_3pose_tensor[10]
        transition_matrix_3pose_tensor_11 = transition_matrix_3pose_tensor[11]
        transition_matrix_3pose_tensor_12 = transition_matrix_3pose_tensor[12]
        transition_matrix_3pose_tensor_13 = transition_matrix_3pose_tensor[13]
        transition_matrix_3pose_tensor_14 = transition_matrix_3pose_tensor[14]
        transition_matrix_3pose_tensor_15 = transition_matrix_3pose_tensor[15]
        transition_matrix_3pose_tensor_16 = transition_matrix_3pose_tensor[16]
        transition_matrix_3pose_tensor_17 = transition_matrix_3pose_tensor[17]
        transition_matrix_3pose_tensor_18 = transition_matrix_3pose_tensor[18]
        transition_matrix_3pose_tensor_19 = transition_matrix_3pose_tensor[19]
        transition_matrix_3pose_tensor_20 = transition_matrix_3pose_tensor[20]
        transition_matrix_3pose_tensor_21 = transition_matrix_3pose_tensor[21]
        transition_matrix_3pose_tensor_22 = transition_matrix_3pose_tensor[22]

    with tf.name_scope('co_3D'):
        # 14 13 12 11 = 10      at angle
        matrix23_14 = tf.matmul(transition_matrix_3pose_tensor_22, transition_matrix_3pose_tensor_13)
        matrix23_14_13 = tf.matmul(matrix23_14, transition_matrix_3pose_tensor_12)
        matrix23_14_13_12 = tf.matmul(matrix23_14_13, transition_matrix_3pose_tensor_11)
        matrix23_14_13_12_11 = tf.matmul(matrix23_14_13_12, transition_matrix_3pose_tensor_10)
        matrix23_14_13_12_11_10 = tf.matmul(matrix23_14_13_12_11, transition_matrix_3pose_tensor_9)
        matrix23_14_13_12_11_10_1 = tf.matmul(matrix23_14_13_12_11_10, transition_matrix_3pose_tensor_0)
        matrix23_14_13_12_11_10_5 = tf.matmul(matrix23_14_13_12_11_10, transition_matrix_3pose_tensor_4)
        matrix23_14_13_12_11_10_5_4 = tf.matmul(matrix23_14_13_12_11_10_5, transition_matrix_3pose_tensor_3)
        matrix23_14_13_12_11_10_5_4_3 = tf.matmul(matrix23_14_13_12_11_10_5_4, transition_matrix_3pose_tensor_2)
        matrix23_14_13_12_11_10_5_4_3_2 = tf.matmul(matrix23_14_13_12_11_10_5_4_3, transition_matrix_3pose_tensor_1)
        matrix23_14_13_12_11_10_6 = tf.matmul(matrix23_14_13_12_11_10, transition_matrix_3pose_tensor_5)
        matrix23_14_13_12_11_10_6_7 = tf.matmul(matrix23_14_13_12_11_10_6, transition_matrix_3pose_tensor_6)
        matrix23_14_13_12_11_10_6_7_8 = tf.matmul(matrix23_14_13_12_11_10_6_7, transition_matrix_3pose_tensor_7)
        matrix23_14_13_12_11_10_6_7_8_9 = tf.matmul(matrix23_14_13_12_11_10_6_7_8, transition_matrix_3pose_tensor_8)

        # abort 15
        matrix23_15 = transition_matrix_3pose_tensor_22
        matrix23_15_17 = tf.matmul(matrix23_15, transition_matrix_3pose_tensor_16)
        matrix23_15_18 = tf.matmul(matrix23_15, transition_matrix_3pose_tensor_17)
        matrix23_15_17_19 = tf.matmul(matrix23_15_17, transition_matrix_3pose_tensor_18)
        matrix23_15_18_20 = tf.matmul(matrix23_15_18, transition_matrix_3pose_tensor_19)
        matrix23_15_17_19_21 = tf.matmul(matrix23_15_17_19, transition_matrix_3pose_tensor_20)
        matrix23_15_18_20_22 = tf.matmul(matrix23_15_18_20, transition_matrix_3pose_tensor_21)
        matrix_sum_sum = tf.stack([matrix23_14_13_12_11_10_1,
                                   matrix23_14_13_12_11_10_1,
                                   matrix23_14_13_12_11_10_1,
                                   matrix23_14_13_12_11_10_5_4_3_2,
                                   matrix23_14_13_12_11_10_5_4_3,
                                   matrix23_14_13_12_11_10_5_4_3,
                                   matrix23_14_13_12_11_10_5_4_3,
                                   matrix23_14_13_12_11_10_5_4_3,
                                   matrix23_14_13_12_11_10_5_4,
                                   matrix23_14_13_12_11_10_5_4,
                                   matrix23_14_13_12_11_10_5_4,
                                   matrix23_14_13_12_11_10_5_4,
                                   matrix23_14_13_12_11_10_5,
                                   matrix23_14_13_12_11_10_5,
                                   matrix23_14_13_12_11_10,
                                   matrix23_14_13_12_11_10_6,
                                   matrix23_14_13_12_11_10_6,
                                   matrix23_14_13_12_11_10_6_7,
                                   matrix23_14_13_12_11_10_6_7,
                                   matrix23_14_13_12_11_10_6_7,
                                   matrix23_14_13_12_11_10_6_7,
                                   matrix23_14_13_12_11_10_6_7_8,
                                   matrix23_14_13_12_11_10_6_7_8,
                                   matrix23_14_13_12_11_10_6_7_8,
                                   matrix23_14_13_12_11_10_6_7_8,
                                   matrix23_14_13_12_11_10_6_7_8_9,
                                   matrix23_14_13_12_11_10,
                                   matrix23_14_13_12_11,
                                   matrix23_14_13_12_11_10,
                                   matrix23_14_13_12_11,
                                   matrix23_14_13_12,
                                   matrix23_14_13_12_11,
                                   matrix23_14_13_12,
                                   matrix23_14_13,
                                   matrix23_14_13_12,
                                   matrix23_14_13,
                                   matrix23_14,
                                   matrix23_14_13,
                                   matrix23_15,
                                   matrix23_15,
                                   matrix23_15,
                                   matrix23_15_17,
                                   matrix23_15_18,
                                   matrix23_15_17,
                                   matrix23_15_18,
                                   matrix23_15_17,
                                   matrix23_15_18,
                                   matrix23_15_17,
                                   matrix23_15_18,
                                   matrix23_15_17_19,
                                   matrix23_15_18_20,
                                   matrix23_15_17_19,
                                   matrix23_15_18_20,
                                   matrix23_15_17_19,
                                   matrix23_15_18_20,
                                   matrix23_15_17_19,
                                   matrix23_15_18_20,
                                   matrix23_15_17_19,
                                   matrix23_15_18_20,
                                   matrix23_15_17_19_21,
                                   matrix23_15_18_20_22,
                                   matrix23_15_17_19_21,
                                   matrix23_15_18_20_22])

        coordinate_3D_tensor = tf.expand_dims(tf.concat([coordinate_Tpose_tensor, tf.ones([63, 1])],
                                                        axis=1),
                                              axis=2, name='co_3D')
        coordinate_final_3D_tensor = tf.matmul(matrix_sum_sum, coordinate_3D_tensor) + tf.tile(tf.expand_dims(root_tensor, axis=0), [63, 1, 1], name='co_f_3D')

    with tf.name_scope('co_2D'):
        M_matrix_tensor = tf.transpose(tf.constant(M_matrix, dtype=TYPE), [2, 0, 1], name='M')
        coordinate_2D_addone_tensor = tf.matmul(tf.tile(tf.expand_dims(M_matrix_tensor, axis=1), [1, 63, 1, 1]),
                                                tf.tile(tf.expand_dims(coordinate_final_3D_tensor, axis=0), [8, 1, 1, 1]), name='co_2D_add')
        coordinate_2D_addone_squeeze_tensor = tf.transpose(tf.squeeze(coordinate_2D_addone_tensor, axis=3), [2, 0, 1], name='co_2D_s_ad')
        coordinate_2D_tensor = tf.stack([coordinate_2D_addone_squeeze_tensor[0] / coordinate_2D_addone_squeeze_tensor[2],
                                          coordinate_2D_addone_squeeze_tensor[1] / coordinate_2D_addone_squeeze_tensor[2]], axis=2, name='co_2D')                                               # 8*63*2

        radius_3D_tensor = tf.tile(radius_tensor, [8, 1], name='radius_3D')
        fl_tensor = tf.tile(tf.transpose(tf.constant(value=fl, dtype=TYPE), [1, 0]), [1, 63], name='fl')
        radius_2D_tensor = fl_tensor * radius_3D_tensor / coordinate_2D_addone_squeeze_tensor[2]

    with tf.name_scope('loss'):
        # todo 优化（数量，平方）
        B_mu_in_tensor = tf.placeholder(shape=[cf.image_cluster_number, 8, 2], dtype=TYPE, name='B_mu_in')
        B_mu_tensor = tf.tile(tf.expand_dims(B_mu_in_tensor, axis=2), [1, 1, 63, 1], name='B_mu')

        B_sigma_in_tensor = tf.placeholder(shape=[cf.image_cluster_number, 8], dtype=TYPE, name='B_sigma_in')
        B_sigma_tensor = tf.tile(tf.expand_dims(B_sigma_in_tensor, axis=2), [1, 1, 63], name='B_sigma')

        D_in_tensor = tf.placeholder(shape=[63, cf.image_cluster_number, 8], dtype=TYPE, name='D_in')
        D_tensor = tf.transpose(D_in_tensor, [1, 2, 0], name='D')

        EII_tensor = tf.placeholder(shape=[cf.image_cluster_number, 8], dtype=TYPE, name='EII')

        B_sigma_square_tensor = tf.square(B_sigma_tensor, name='B_sig_sq')
        radius_2D_square_tensor = tf.tile(tf.expand_dims(tf.square(radius_2D_tensor) ,axis=0), [cf.image_cluster_number, 1, 1], name='ra_2D_sq')

        loss_31 = -tf.square(tf.norm(tf.tile(tf.expand_dims(coordinate_2D_tensor ,axis=0), [cf.image_cluster_number, 1, 1, 1]) - B_mu_tensor, axis=3), name='loss_31')
        loss_3 = tf.exp(loss_31 / (radius_2D_square_tensor + B_sigma_square_tensor), name='loss_3')
        loss_2 = B_sigma_square_tensor * radius_2D_square_tensor / (B_sigma_square_tensor + radius_2D_square_tensor)
        loss_0 = D_tensor * loss_2 * loss_3

        loss_theta  = tf.cast(tf.logical_or(euler_tensor > rc.pose_up,
                                            euler_tensor < rc.pose_low),
                              tf.float32) * \
                      tf.minimum(tf.pow(euler_tensor - rc.pose_up, 2),
                                 tf.pow(euler_tensor - rc.pose_low, 2))
        loss_theta_scalar = tf.reduce_sum(loss_theta)

        if 1:
            loss_1_26 = loss_0[:, :, 0:26]
            loss_27_41 = loss_0[:, :, 26:41] * 2
            loss_42_63 = loss_0[:, :, 41:63]
            loss_0 = tf.concat([loss_1_26, loss_27_41, loss_42_63], axis=2)

        loss_model_view8_eachCluster = -tf.minimum(tf.reduce_sum(loss_0, axis=2) * 2 * math.pi, EII_tensor)
        loss_model_view8 = tf.reduce_sum(loss_model_view8_eachCluster, axis=0, name='loss_view8')
        # loss_model = tf.reduce_sum(loss_model_view8, name='loss')
        loss_model = loss_model_view8[0] + loss_model_view8[1] + loss_model_view8[2] + loss_model_view8[3] + \
                     loss_model_view8[4] + loss_model_view8[5] + loss_model_view8[6] + loss_model_view8[7]

        loss = loss_theta_scalar + loss_model

        loss_model_view8_eachBall = -tf.reduce_sum(loss_0, axis=0)

        if 1:
            tf.summary.scalar('loss_result', loss)
        if 0:
            tf.summary.scalar('loss_theta_scalar', loss_theta_scalar)
            tf.summary.scalar('loss_model', loss_model)
        for i_view in range(8):
            if 0:
                tf.summary.scalar('loss_v%d' % i_view, loss_model_view8[i_view])
            if 0:
                for i_body in range(cf.bodyNumBall):
                    tf.summary.scalar('loss_Ball_%d_v_%d' % (i_body,i_view), loss_model_view8_eachBall[i_view][i_body])


    with tf.name_scope('train'):
        train_optimizer = tf.train.GradientDescentOptimizer(train_step).minimize(loss, gate_gradients=0)

    init = tf.global_variables_initializer()
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    with tf.Session() as sess:
        sess.run(init)

        merged = tf.summary.merge_all()
        if switch_writer == True:
            writer = tf.summary.FileWriter(log_path, sess.graph)
        else:
            writer = tf.summary.FileWriter(log_path)

        if if_save_time == True:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()


        B_mu_npz = []
        B_sigma_npz = []
        D_npz = []
        EII_npz = []
        getImageForColor = []

        time_frame = datetime.datetime.now()
        frame_end = frame_start + num_frames
        for i_frame in range(frame_start, frame_end):

            # new frame
            print("\n")

            B_mu_npz.clear()
            B_sigma_npz.clear()
            D_npz.clear()
            EII_npz.clear()
            getImageForColor.clear()

            getImageClustered = []

            if 0:
                pass
                # myPreProduce.getImageProduced(B_mu_new_fs=B_mu_npz, B_sigma_new_fs=B_sigma_npz,
                #                               D_new_fs=D_npz, EII_new_fs=EII_npz,
                #                               i_frame=i_frame, video_choose=video_choose,
                #                               ifAImage=not ifAllVideo, ifShape=ifShape, mode=mode,
                #                               path_video=path_video, path_Tpose=path_Tpose, path_TposeColor=path_TposeColor,
                #                               getImageForColor=getImageForColor, getImageClustered=getImageClustered, max_of_cluster=max_of_cluster)
                #
            else:
                mu_sigma_D_EII = [B_mu_npz, B_sigma_npz, D_npz, EII_npz]
                image_color_clustered = [getImageForColor, getImageClustered]
                if ifShape:
                    shape_or_pose = "shape"
                else:
                    shape_or_pose = "pose"
                mode_config = ModeConfig(i_frame=i_frame, ifAImage=not ifAllVideo, shape_or_pose=ifShape, mode=mode,
                                         path_video=path_video, path_Tpose=path_Tpose, path_TposeColor=path_TposeColor)
                mImage_produce.getImageProduced(mu_sigma_D_EII, image_color_clustered, max_of_cluster, mode_config)

            def save_result(filename_save):
                axisAngle_3_23_final = np.float64(sess.run(axisAngle_3_23_tensor))
                root_final = np.float64(sess.run(root_variable))
                pose_save = np.concatenate((axisAngle_3_23_final, root_final), axis=1)

                SaveAxisAngle(path_save=path_save, axisAngle_3_24=pose_save, filename_save=filename_save)

            time_delta = 0
            # run
            if mode == "pose_in_all_video":
                i_begin = train_times * i_frame
                i_end = train_times * (i_frame + 1)
            else:
                i_begin = train_times * i_frame
                i_end = train_times * (i_frame + 1) + 1
            flag_fisrt = True
            for i in range(i_begin, i_end):
                # log
                if i % (10) == 0 and 0:
                    if if_save_time == True:
                        _, result = sess.run([train_optimizer, merged], feed_dict={
                        B_mu_in_tensor: B_mu_npz[0],
                        B_sigma_in_tensor: B_sigma_npz[0],
                        D_in_tensor: D_npz[0],
                        EII_tensor: EII_npz[0]}, options=run_options, run_metadata=run_metadata)
                        writer.add_summary(result, i)
                        writer.add_run_metadata(run_metadata, 'step{}'.format(i))
                    else:
                        _, result = sess.run([train_optimizer, merged], feed_dict={
                        B_mu_in_tensor: B_mu_npz[0],
                        B_sigma_in_tensor: B_sigma_npz[0],
                        D_in_tensor: D_npz[0],
                        EII_tensor: EII_npz[0]})
                        writer.add_summary(result, i - 1.8 * 1000000000)
                # print info. 运行时间：500/55s
                times_per_print = 500
                if i % (times_per_print) == 0 and 1:
                    if flag_fisrt == False:
                        if time_delta == 0:
                            time_delta = datetime.datetime.now() - print_now
                        else:
                            time_delta = 0.6 * time_delta + 0.4 * (datetime.datetime.now() - print_now)
                    print_now = datetime.datetime.now()
                    flag_fisrt = False

                    print("frame: %d/%d, i_times: %d, "
                          "当前运行时间：%s, "
                          "当前运行次数：%d/%d" %
                          (i_frame, frame_end, i_times,
                           str(datetime.datetime.now() - startTime_system),
                           i % train_times, train_times), end="    ")
                    print("时间间隔：{0}".format(str(time_delta)), end="    ")
                    print("当前帧预计剩余时间：{0}".format(str(time_delta * (train_times - (i % train_times)) / times_per_print)))

                # save image
                if 0 and (i + 1) % cf.trainTimes_poseInPose == 0 \
                        or mode == "expect color" \
                        or (mode == "sil_for_deep" and ((i + 1) % cf.trainTimes_poseInShape_array[i_times]) == 0 and 0):
                    coordinate_2D_eval = np.float64(sess.run(coordinate_2D_tensor))
                    radius_2D_eval = np.float64(sess.run(radius_2D_tensor))

                    if ifAllVideo == False:
                        path_color_for_drawBody = cf.path_TposeColor
                    elif ifAllVideo == True:
                        # path_color_for_drawBody = pathFile_TposeColor_optimized
                        path_color_for_drawBody = path_TposeColor + ".mat"

                    if mode== "pose_in_all_video":
                        note_in_saved_image = "t%d_pose%d_i%d" % (i_frame, i_poseth, i % max(i_begin, 1))
                    else:
                        note_in_saved_image = "t%d_pose%d_i%d" % (i_times, i_poseth, i % max(i_begin, 1))

                    # "cluster, origin"
                    class CF_for_drawBody():
                        def __init__(self, mode_image="origin"):
                            self.mode = "circle"
                            self.mode_image = mode_image
                            self.note = note_in_saved_image
                            self.i_frame = i_frame
                            self.windowName = ""
                            self.pInModeCS = []


                    data_result = [coordinate_2D_eval,
                                   radius_2D_eval,
                                   read_bodyColor(path_color_for_drawBody)]
                    if frame_start >= 150:
                        cf_for_drawBody = CF_for_drawBody(mode_image="origin")
                        cf_for_drawBody = CF_for_drawBody(mode_image="cluster")
                        draw_body2(getImageClustered, data_result, cf_for_drawBody)
                        # draw_body2(None, data_result, cf_for_drawBody)
                    else:
                        cf_for_drawBody = CF_for_drawBody(mode_image="cluster")
                        draw_body(mode=cf_for_drawBody.mode,
                                  mode_image=cf_for_drawBody.mode_image,
                                  coordinate_2D=data_result[0],
                                  radius_2D=data_result[1],
                                  color=data_result[2],
                                  windowName="times: " + str(i), note=note_in_saved_image,
                                  imageClustered=getImageClustered)

                # run
                sess.run(train_optimizer, feed_dict={
                        B_mu_in_tensor: B_mu_npz[0],
                        B_sigma_in_tensor: B_sigma_npz[0],
                        D_in_tensor: D_npz[0],
                        EII_tensor: EII_npz[0]})

                if i % (10) == 0 and 0:
                    if ifAllVideo == True:
                        filename_save_real = filename_save + ("%d-i%d.mat" % (i_frame, i))
                    save_result(filename_save_real)

            # save
            if if_save_result == True:
                if ifAllVideo == True:
                    filename_save_real = filename_save + str(i_frame) + ".mat"
                save_result(filename_save_real)

            time_frame_last = time_frame
            time_frame = datetime.datetime.now()
            time_frame_delta= time_frame - time_frame_last
            print("frame: %d/%d    "
                  "当前运行时间：%s，"
                  "time per frame: %s, time of remainder frame: %s" %
                  (i_frame, num_frames,
                   str(time_frame - startTime_system),
                   str(time_frame_delta), str(time_frame_delta * (num_frames - (i_frame - frame_start) - 1))))

    tf.reset_default_graph()


def run2_forShape(ifPose = True, mode=None, filename_pose1="", filename_pose2="", filename_rl="",
                  filename_save="", note_out="", filename_save_color="",
                  train_times=2000, i_times=0, modeConfig={}):

    """mode: sil_for_deep, expect color"""

    # ifPose = False
    ifShape = not ifPose

    ifAllVideo = False

    # set..............................................................................................................

    path_video_1_forColor = cf.path_imageCol_1
    path_video_2_forColor = cf.path_imageCol_2

    if mode == "optimize_shape_inRGB":
        path_video_1 = cf.pathVideo_
        path_video_2 = cf.pathVideo_
    else:
        path_video_1 = cf.path_imageSil_1
        path_video_2 = cf.path_imageSil_2

    path_rl = filename_rl

    path_save = cf.pathSave



    video_choose = ' '

    # 输出注释
    note_simple = "_3e8_2k" + note_out



    path_pose_init1 = cf.pathSave + \
                      cf.str_dir_final + filename_pose1
    path_pose_init2 = cf.pathSave + \
                      cf.str_dir_final + filename_pose2


    # 不可随意修改，初始姿态
    frame_start = 0
    frame_start = frame_start
    # todo string
    num_frames = 1

    # canshu............................................................................................................

    if ifShape ==True:
        train_step = 0.00000001 * 30 * 10 * 10
        filename_color = filename_save[0:10] + "_color_" + filename_save[17:24]
    else:
        train_step = 0
        print("No Trainable Variables")
        exit(1)


    # switch
    switch_writer = False
    if_save_time = False
    if switch_writer == False:
        if_save_time = False

    if_save_result = True  # no influence on time

    # preProduce................................................................................................................
    note = "_" + video_choose + note_simple

    # filename_image = '0 data_image/imageNew_' + video_choose + '.mat'

    begin = datetime.datetime.now()


    if if_save_result == True:
        mkdir(path_save)

    path_Tpose = cf.pathFile_Tpose
    if 1:
        path_bodyColor = cf.path_TposeColor
    else:
        path_bodyColor = cf.path_save_direction + "181023_3_color_pose0.mat"

    path_M_matrix = cf.pathCameraParams

    log_path = cf.path_log + cf.str_dir_level_12 + begin.strftime('%m%d') + begin.strftime('_%H%M') + note


    TYPE = tf.float32

    # data
    joint = rc.joint

    # main..................................................................................................................

    M_matrix, fl = read_M_matrix(path_M_matrix)

    with tf.name_scope('body'):
        rl = read_rl(path_rl=path_rl)
        if mode == "optimize_shape_inRGB":
            trainable_r = False
        else:
            trainable_r = True
        trainable_l = ifShape
        r_tensor = tf.Variable(rl[0:1, 0:36], dtype=TYPE, trainable=trainable_r, name="r")
        l_tensor = tf.Variable(rl[0:1, 36:52], dtype=TYPE, trainable=trainable_l, name="l")

        # rl_tensor = tf.Variable(rl, dtype=TYPE, trainable=ifShape, name="rl")
        rl_tensor = tf.concat([r_tensor, l_tensor], axis=1)
        Tpose = read_Tpose(path_Tpose=path_Tpose, rl_tensor=rl_tensor)
        radius_tensor = Tpose[0:1, 0:63]                                        # 1*63
        # coordinate_Tpose = Tpose[0][63:252][np.newaxis, :].reshape(3, 63, order='F')     # 3*63
        coordinate_Tpose_tensor = tf.reshape(Tpose[0:1, 63:252], shape=[63, 3])

        for i_pose69 in range(0, 69):
            i_pose = i_pose69 // 3      # 第几个关节
            j_pose = i_pose69 % 3       # 第几个
            temp2 = joint[i_pose]       # 旋转节点坐标
            # todo 前后优化
            names["T_vector_old_co_" + str(j_pose) + "_" + str(i_pose) + "_tensor"] = coordinate_Tpose_tensor[joint[i_pose] - 1, 0:3]
        for j_pose in range(3):
            names["T_vector_old_co_" + str(j_pose) + "_tensor"] = tf.stack([names["T_vector_old_co_" + str(j_pose) + "_" + str(0) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(1) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(2) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(3) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(4) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(5) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(6) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(7) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(8) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(9) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(10) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(11) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(12) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(13) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(14) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(15) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(16) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(17) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(18) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(19) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(20) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(21) + "_tensor"],
                                                                       names["T_vector_old_co_" + str(j_pose) + "_" + str(22) + "_tensor"]], axis=0)
        T_vector_old_co_tensor = tf.stack([names["T_vector_old_co_" + str(0) + "_tensor"],
                                           names["T_vector_old_co_" + str(1) + "_tensor"],
                                           names["T_vector_old_co_" + str(2) + "_tensor"]], axis=0)

    with tf.name_scope('pose'):
        axisAngle_3_24_init1 = read_axisAngle_init(path_pose_init=path_pose_init1)
        axisAngle_3_24_init2 = read_axisAngle_init(path_pose_init=path_pose_init2)
        axisAngle_3_24_init = np.stack([axisAngle_3_24_init1, axisAngle_3_24_init2], axis=0)
        axisAngle_3_20_init = np.concatenate([
            axisAngle_3_24_init[:, :, 0:10],
            axisAngle_3_24_init[:, :, 14:24]], axis=2)

        # axisAngle_3_23_tensor = tf.Variable(initial_value=axisAngle_3_24_init[:, 0:23],
        #                           name='axisAngle', dtype=TYPE,
        #                           trainable=ifPose)

        axisAngle_3_1_9_tensor = tf.Variable(initial_value=axisAngle_3_20_init[:, :, 0:9],
                                            name='axisAngle1', dtype=TYPE,
                                            trainable=ifPose)
        axisAngle_3_10_10_tensor = tf.Variable(initial_value=axisAngle_3_20_init[:, :, 9:10] * 5.,
                                            name='axisAngle2', dtype=TYPE,
                                            trainable=ifPose) / 5.
        axisAngle_3_11_19_tensor = tf.Variable(initial_value=axisAngle_3_20_init[:, :, 10:19],
                                            name='axisAngle3', dtype=TYPE,
                                            trainable=ifPose)

        # axisAngle_3_19_tensor = tf.Variable(initial_value=axisAngle_3_20_init[:, 0:19],
        #                           name='axisAngle', dtype=TYPE,
        #                           trainable=ifPose)
        axisAngle_3_19_tensor = tf.concat([axisAngle_3_1_9_tensor, axisAngle_3_10_10_tensor, axisAngle_3_11_19_tensor], axis=2)
        axisAngle_3_23_tensor = tf.concat([axisAngle_3_19_tensor[:, :, 0:10],
                                           axisAngle_3_19_tensor[:, :, 9:10],
                                           axisAngle_3_19_tensor[:, :, 9:10],
                                           axisAngle_3_19_tensor[:, :, 9:10],
                                           axisAngle_3_19_tensor[:, :, 9:10],
                                           axisAngle_3_19_tensor[:, :, 10:19]], axis=2)
        root_variable = tf.multiply(
            tf.Variable(initial_value=axisAngle_3_24_init[:, :, 23:24] / 173., dtype=TYPE, name='root/173.', trainable=ifPose), 173.,
                                    name='root_31')
        root_tensor = tf.concat([root_variable, tf.zeros(shape=[2, 1, 1])], axis=1, name='root_41')

        #
        theta = tf.sqrt(tf.multiply(axisAngle_3_23_tensor[:, 0:1, 0:23], axisAngle_3_23_tensor[:, 0:1, 0:23]) +
                        tf.multiply(axisAngle_3_23_tensor[:, 1:2, 0:23], axisAngle_3_23_tensor[:, 1:2, 0:23]) +
                        tf.multiply(axisAngle_3_23_tensor[:, 2:3, 0:23], axisAngle_3_23_tensor[:, 2:3, 0:23]) + 1e-9)

        axisAngle_4_23_axis = axisAngle_3_23_tensor / (theta)

    with tf.name_scope('rota_ma'):
        # todo test
        zeros_23 = tf.zeros(shape=[2, 23])
        ones_23 = tf.ones(shape=[2, 23])

        wx_23_1_3_first =  tf.stack([                   zeros_23, -axisAngle_4_23_axis[:, 3 - 1],  axisAngle_4_23_axis[:, 2 - 1]], axis=2, name='wx_23_1_3_first')
        wx_23_1_3_second = tf.stack([ axisAngle_4_23_axis[:, 3 - 1],                    zeros_23, -axisAngle_4_23_axis[:, 1 - 1]], axis=2, name='wx_23_1_3_second')
        wx_23_1_3_third =  tf.stack([-axisAngle_4_23_axis[:, 2 - 1],  axisAngle_4_23_axis[:, 1 - 1],                   zeros_23,], axis=2, name='wx_23_1_3_third')
        # wx_23_3_3 = tf.stack([wx_23_1_3_first, wx_23_1_3_second, wx_23_1_3_third], axis=2, name='wx_23_3_3')
        wx_23_3_3 = tf.stack([wx_23_1_3_first, wx_23_1_3_second, wx_23_1_3_third], axis=2, name='wx_23_3_3')

        diag = tf.tile(tf.expand_dims(tf.diag([1., 1., 1.]), axis=0), multiples=[23, 1, 1], name='diag')
        diag = tf.stack([diag, diag], axis=0)
        dlt = 1
        R_23_3_3 = diag + wx_23_3_3 * tf.tile(tf.expand_dims(tf.expand_dims(tf.sin(theta[:, 0]), axis=2), axis=2), [1, 1, 3, 3]) + \
                   tf.matmul(wx_23_3_3, wx_23_3_3) * tf.tile(tf.expand_dims(tf.expand_dims(1 - tf.cos(theta[:, 0]), axis=2), axis=2), [1, 1, 3, 3])


        #
        # rotation_matrix_0_0 = tf.stack([ones_23, zeros_23, zeros_23], axis=1, name='rota_0_0')
        # rotation_matrix_0_1 = tf.stack(
        #     [zeros_23, pose_3_23_tensor_cos_0, -pose_3_23_tensor_sin_0], axis=1, name='rota_0_1')
        # rotation_matrix_0_2 = tf.stack(
        #     [zeros_23, pose_3_23_tensor_sin_0, pose_3_23_tensor_cos_0], axis=1, name='rota_0_2')
        # rotation_matrix_0_sum = tf.stack([rotation_matrix_0_0, rotation_matrix_0_1, rotation_matrix_0_2], axis=1, name='rota_0_sum')
        #
        # rotation_matrix_1_0 = tf.stack(
        #     [pose_3_23_tensor_cos_1, zeros_23, pose_3_23_tensor_sin_1], axis=1, name='rota_1_0')
        # rotation_matrix_1_1 = tf.stack([zeros_23, ones_23, zeros_23], axis=1, name='rota_1_1')
        # rotation_matrix_1_2 = tf.stack(
        #     [-pose_3_23_tensor_sin_1, zeros_23, pose_3_23_tensor_cos_1], axis=1, name='rota_1_2')
        # rotation_matrix_1_sum = tf.stack([rotation_matrix_1_0, rotation_matrix_1_1, rotation_matrix_1_2], axis=1, name='rota_1_sum')
        #
        # rotation_matrix_2_0 = tf.stack(
        #     [pose_3_23_tensor_cos_2, -pose_3_23_tensor_sin_2, zeros_23], axis=1, name='rota_2_0')
        # rotation_matrix_2_1 = tf.stack(
        #     [pose_3_23_tensor_sin_2, pose_3_23_tensor_cos_2, zeros_23], axis=1, name='rota_2_1')
        # rotation_matrix_2_2 = tf.stack([zeros_23, zeros_23, ones_23], axis=1, name='rota_2_2')
        # rotation_matrix_2_sum = tf.stack([rotation_matrix_2_0, rotation_matrix_2_1, rotation_matrix_2_2], axis=1, name='rota_2_sum')
        #
        # # 3 * 23 * 3 * 3
        # rotation_matrix_sum_sum = tf.stack([rotation_matrix_0_sum, rotation_matrix_1_sum, rotation_matrix_2_sum], axis=0, name='rota_s_sum')

    with tf.name_scope('T_vec'):
            coordinate_joint_tensor = tf.expand_dims(T_vector_old_co_tensor[0, :, :], axis=2, name='co_joint')
            coordinate_joint_tensor = tf.stack([coordinate_joint_tensor, coordinate_joint_tensor], axis=0)
            diag_23_3_3 = tf.tile(tf.expand_dims(tf.diag([1., 1., 1.]), axis=0), multiples=[23, 1, 1], name='diag')
            diag_23_3_3 = tf.stack([diag_23_3_3, diag_23_3_3], axis=0)
            temp = (diag_23_3_3 - R_23_3_3)
            T_vector_tensor = tf.matmul(temp, coordinate_joint_tensor, name='T_vector')


    with tf.name_scope('tran_ma'):
        zero3one1 = tf.concat([tf.zeros(shape=[2, 23, 1, 3]), tf.ones(shape=[2, 23, 1, 1])], axis=3)
        transition_matrix_tensor = tf.concat([tf.concat([R_23_3_3, T_vector_tensor], axis=3), zero3one1], axis=2, name='transi')
        transition_matrix_3pose_tensor = transition_matrix_tensor

        transition_matrix_3pose_tensor_0 = transition_matrix_3pose_tensor[:, 0]
        transition_matrix_3pose_tensor_1 = transition_matrix_3pose_tensor[:, 1]
        transition_matrix_3pose_tensor_2 = transition_matrix_3pose_tensor[:, 2]
        transition_matrix_3pose_tensor_3 = transition_matrix_3pose_tensor[:, 3]
        transition_matrix_3pose_tensor_4 = transition_matrix_3pose_tensor[:, 4]
        transition_matrix_3pose_tensor_5 = transition_matrix_3pose_tensor[:, 5]
        transition_matrix_3pose_tensor_6 = transition_matrix_3pose_tensor[:, 6]
        transition_matrix_3pose_tensor_7 = transition_matrix_3pose_tensor[:, 7]
        transition_matrix_3pose_tensor_8 = transition_matrix_3pose_tensor[:, 8]
        transition_matrix_3pose_tensor_9 = transition_matrix_3pose_tensor[:, 9]
        transition_matrix_3pose_tensor_10 = transition_matrix_3pose_tensor[:, 10]
        transition_matrix_3pose_tensor_11 = transition_matrix_3pose_tensor[:, 11]
        transition_matrix_3pose_tensor_12 = transition_matrix_3pose_tensor[:, 12]
        transition_matrix_3pose_tensor_13 = transition_matrix_3pose_tensor[:, 13]
        transition_matrix_3pose_tensor_14 = transition_matrix_3pose_tensor[:, 14]
        transition_matrix_3pose_tensor_15 = transition_matrix_3pose_tensor[:, 15]
        transition_matrix_3pose_tensor_16 = transition_matrix_3pose_tensor[:, 16]
        transition_matrix_3pose_tensor_17 = transition_matrix_3pose_tensor[:, 17]
        transition_matrix_3pose_tensor_18 = transition_matrix_3pose_tensor[:, 18]
        transition_matrix_3pose_tensor_19 = transition_matrix_3pose_tensor[:, 19]
        transition_matrix_3pose_tensor_20 = transition_matrix_3pose_tensor[:, 20]
        transition_matrix_3pose_tensor_21 = transition_matrix_3pose_tensor[:, 21]
        transition_matrix_3pose_tensor_22 = transition_matrix_3pose_tensor[:, 22]

    with tf.name_scope('co_3D'):
        # 14 13 12 11 = 10      at angle
        matrix23_14 = tf.matmul(transition_matrix_3pose_tensor_22, transition_matrix_3pose_tensor_13)
        matrix23_14_13 = tf.matmul(matrix23_14, transition_matrix_3pose_tensor_12)
        matrix23_14_13_12 = tf.matmul(matrix23_14_13, transition_matrix_3pose_tensor_11)
        matrix23_14_13_12_11 = tf.matmul(matrix23_14_13_12, transition_matrix_3pose_tensor_10)
        matrix23_14_13_12_11_10 = tf.matmul(matrix23_14_13_12_11, transition_matrix_3pose_tensor_9)
        matrix23_14_13_12_11_10_1 = tf.matmul(matrix23_14_13_12_11_10, transition_matrix_3pose_tensor_0)
        matrix23_14_13_12_11_10_5 = tf.matmul(matrix23_14_13_12_11_10, transition_matrix_3pose_tensor_4)
        matrix23_14_13_12_11_10_5_4 = tf.matmul(matrix23_14_13_12_11_10_5, transition_matrix_3pose_tensor_3)
        matrix23_14_13_12_11_10_5_4_3 = tf.matmul(matrix23_14_13_12_11_10_5_4, transition_matrix_3pose_tensor_2)
        matrix23_14_13_12_11_10_5_4_3_2 = tf.matmul(matrix23_14_13_12_11_10_5_4_3, transition_matrix_3pose_tensor_1)
        matrix23_14_13_12_11_10_6 = tf.matmul(matrix23_14_13_12_11_10, transition_matrix_3pose_tensor_5)
        matrix23_14_13_12_11_10_6_7 = tf.matmul(matrix23_14_13_12_11_10_6, transition_matrix_3pose_tensor_6)
        matrix23_14_13_12_11_10_6_7_8 = tf.matmul(matrix23_14_13_12_11_10_6_7, transition_matrix_3pose_tensor_7)
        matrix23_14_13_12_11_10_6_7_8_9 = tf.matmul(matrix23_14_13_12_11_10_6_7_8, transition_matrix_3pose_tensor_8)

        # abort 15
        matrix23_15 = transition_matrix_3pose_tensor_22
        matrix23_15_17 = tf.matmul(matrix23_15, transition_matrix_3pose_tensor_16)
        matrix23_15_18 = tf.matmul(matrix23_15, transition_matrix_3pose_tensor_17)
        matrix23_15_17_19 = tf.matmul(matrix23_15_17, transition_matrix_3pose_tensor_18)
        matrix23_15_18_20 = tf.matmul(matrix23_15_18, transition_matrix_3pose_tensor_19)
        matrix23_15_17_19_21 = tf.matmul(matrix23_15_17_19, transition_matrix_3pose_tensor_20)
        matrix23_15_18_20_22 = tf.matmul(matrix23_15_18_20, transition_matrix_3pose_tensor_21)
        matrix_sum_sum = tf.stack([matrix23_14_13_12_11_10_1,
                                   matrix23_14_13_12_11_10_1,
                                   matrix23_14_13_12_11_10_1,
                                   matrix23_14_13_12_11_10_5_4_3_2,
                                   matrix23_14_13_12_11_10_5_4_3,
                                   matrix23_14_13_12_11_10_5_4_3,
                                   matrix23_14_13_12_11_10_5_4_3,
                                   matrix23_14_13_12_11_10_5_4_3,
                                   matrix23_14_13_12_11_10_5_4,
                                   matrix23_14_13_12_11_10_5_4,
                                   matrix23_14_13_12_11_10_5_4,
                                   matrix23_14_13_12_11_10_5_4,
                                   matrix23_14_13_12_11_10_5,
                                   matrix23_14_13_12_11_10_5,
                                   matrix23_14_13_12_11_10,
                                   matrix23_14_13_12_11_10_6,
                                   matrix23_14_13_12_11_10_6,
                                   matrix23_14_13_12_11_10_6_7,
                                   matrix23_14_13_12_11_10_6_7,
                                   matrix23_14_13_12_11_10_6_7,
                                   matrix23_14_13_12_11_10_6_7,
                                   matrix23_14_13_12_11_10_6_7_8,
                                   matrix23_14_13_12_11_10_6_7_8,
                                   matrix23_14_13_12_11_10_6_7_8,
                                   matrix23_14_13_12_11_10_6_7_8,
                                   matrix23_14_13_12_11_10_6_7_8_9,
                                   matrix23_14_13_12_11_10,
                                   matrix23_14_13_12_11,
                                   matrix23_14_13_12_11_10,
                                   matrix23_14_13_12_11,
                                   matrix23_14_13_12,
                                   matrix23_14_13_12_11,
                                   matrix23_14_13_12,
                                   matrix23_14_13,
                                   matrix23_14_13_12,
                                   matrix23_14_13,
                                   matrix23_14,
                                   matrix23_14_13,
                                   matrix23_15,
                                   matrix23_15,
                                   matrix23_15,
                                   matrix23_15_17,
                                   matrix23_15_18,
                                   matrix23_15_17,
                                   matrix23_15_18,
                                   matrix23_15_17,
                                   matrix23_15_18,
                                   matrix23_15_17,
                                   matrix23_15_18,
                                   matrix23_15_17_19,
                                   matrix23_15_18_20,
                                   matrix23_15_17_19,
                                   matrix23_15_18_20,
                                   matrix23_15_17_19,
                                   matrix23_15_18_20,
                                   matrix23_15_17_19,
                                   matrix23_15_18_20,
                                   matrix23_15_17_19,
                                   matrix23_15_18_20,
                                   matrix23_15_17_19_21,
                                   matrix23_15_18_20_22,
                                   matrix23_15_17_19_21,
                                   matrix23_15_18_20_22], axis=1)
        coordinate_Tpose_tensor = tf.stack([coordinate_Tpose_tensor, coordinate_Tpose_tensor], axis=0)
        coordinate_3D_tensor = tf.expand_dims(tf.concat([coordinate_Tpose_tensor, tf.ones([2, 63, 1])],
                                                        axis=2),
                                              axis=3, name='co_3D')
        coordinate_final_3D_tensor = tf.matmul(matrix_sum_sum, coordinate_3D_tensor) + tf.tile(tf.expand_dims(root_tensor, axis=1), [1, 63, 1, 1], name='co_f_3D')

    with tf.name_scope('co_2D'):
        M_matrix_tensor = tf.transpose(tf.constant(M_matrix, dtype=TYPE), [2, 0, 1], name='M')
        M_matrix_tensor = tf.stack([M_matrix_tensor, M_matrix_tensor], axis=0)
        coordinate_2D_addone_tensor = tf.matmul(tf.tile(tf.expand_dims(M_matrix_tensor, axis=2), [1, 1, 63, 1, 1]),
                                                tf.tile(tf.expand_dims(coordinate_final_3D_tensor, axis=1), [1, 8, 1, 1, 1]), name='co_2D_add')
        coordinate_2D_addone_squeeze_tensor = tf.transpose(tf.squeeze(coordinate_2D_addone_tensor, axis=4), [0, 3, 1, 2], name='co_2D_s_ad')
        coordinate_2D_tensor = tf.stack([coordinate_2D_addone_squeeze_tensor[:, 0] / coordinate_2D_addone_squeeze_tensor[:, 2],
                                          coordinate_2D_addone_squeeze_tensor[:, 1] / coordinate_2D_addone_squeeze_tensor[:, 2]], axis=3, name='co_2D')                                               # 8*63*2

        radius_tensor = tf.stack([radius_tensor, radius_tensor], axis=0)
        radius_3D_tensor = tf.tile(radius_tensor, [1, 8, 1], name='radius_3D')
        fl_tensor = tf.tile(tf.transpose(tf.constant(value=fl, dtype=TYPE), [1, 0]), [1, 63], name='fl')
        fl_tensor = tf.stack([fl_tensor, fl_tensor], axis=0)
        radius_2D_tensor = fl_tensor * radius_3D_tensor / coordinate_2D_addone_squeeze_tensor[:, 2]

    with tf.name_scope('loss'):
            # todo 优化（数量，平方）
            B_mu_in_tensor = tf.placeholder(shape=[2, cf.image_cluster_number, 8, 2], dtype=TYPE, name='B_mu_in')
            B_mu_tensor = tf.tile(tf.expand_dims(B_mu_in_tensor, axis=3), [1, 1, 1, 63, 1], name='B_mu')

            B_sigma_in_tensor = tf.placeholder(shape=[2, cf.image_cluster_number, 8], dtype=TYPE, name='B_sigma_in')
            B_sigma_tensor = tf.tile(tf.expand_dims(B_sigma_in_tensor, axis=3), [1, 1, 1, 63], name='B_sigma')

            D_in_tensor = tf.placeholder(shape=[2, 63, cf.image_cluster_number, 8], dtype=TYPE, name='D_in')
            D_tensor = tf.transpose(D_in_tensor, [0, 2, 3, 1], name='D')

            EII_tensor = tf.placeholder(shape=[2, cf.image_cluster_number, 8], dtype=TYPE, name='EII')

            B_sigma_square_tensor = tf.square(B_sigma_tensor, name='B_sig_sq')
            radius_2D_square_tensor = tf.tile(tf.expand_dims(tf.square(radius_2D_tensor) ,axis=1), [1, cf.image_cluster_number, 1, 1], name='ra_2D_sq')

            loss_31 = -tf.square(tf.norm(tf.tile(tf.expand_dims(coordinate_2D_tensor ,axis=1), [1, cf.image_cluster_number, 1, 1, 1]) - B_mu_tensor, axis=4), name='loss_31')
            loss_3 = tf.exp(loss_31 / (radius_2D_square_tensor + B_sigma_square_tensor), name='loss_3')
            loss_2 = B_sigma_square_tensor * radius_2D_square_tensor / (B_sigma_square_tensor + radius_2D_square_tensor)
            loss_0 = D_tensor * loss_2 * loss_3

            loss = -tf.reduce_sum(tf.minimum(tf.reduce_sum(loss_0, axis=3) * 2 * math.pi, EII_tensor), name='loss')
            # todo
            # tf.summary.scalar('loss', names['loss'])
            tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train_optimizer = tf.train.GradientDescentOptimizer(train_step).minimize(loss, gate_gradients=0)

    init = tf.global_variables_initializer()
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    with tf.Session() as sess:
        sess.run(init)

        merged = tf.summary.merge_all()
        if switch_writer == True:
            writer = tf.summary.FileWriter(log_path, sess.graph)
        else:
            writer = tf.summary.FileWriter(log_path)

        if if_save_time == True:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        B_mu_npz = []
        B_sigma_npz = []
        D_npz = []
        EII_npz = []

        B_mu_npz_2 = []
        B_sigma_npz_2 = []
        D_npz_2 = []
        EII_npz_2 = []

        getImageForColor_1 = []
        getImageForColor_2 = []

        # todo useless for shape
        for i_frame in range(frame_start, num_frames):
            # print(sess.run(rl_tensor))
            B_mu_npz.clear()
            B_sigma_npz.clear()
            D_npz.clear()
            EII_npz.clear()

            B_mu_npz_2.clear()
            B_sigma_npz_2.clear()
            D_npz_2.clear()
            EII_npz_2.clear()

            getImageForColor_1.clear()
            getImageForColor_2.clear()

            getImageClustered_1 = []
            getImageClustered_2 = []
            # if modeConfig != None:
            modeConfig_in_getImageProduced = modeConfig.copy()
            modeConfig_in_getImageProduced['id'] = 0
            myPreProduce.getImageProduced(B_mu_new_fs=B_mu_npz, B_sigma_new_fs=B_sigma_npz,
                                          D_new_fs=D_npz, EII_new_fs=EII_npz,
                                          i_frame=i_frame, video_choose=video_choose,
                                          ifAImage=not ifAllVideo, ifShape=True, mode=mode,
                                          path_video=path_video_1, path_Tpose=path_Tpose, path_TposeColor=path_bodyColor,
                                          getImageForColor=getImageForColor_1, path_video_forColor=path_video_1_forColor,
                                          getImageClustered=getImageClustered_1, modeConfig=modeConfig_in_getImageProduced)
            modeConfig_in_getImageProduced['id'] = 1
            myPreProduce.getImageProduced(B_mu_new_fs=B_mu_npz_2, B_sigma_new_fs=B_sigma_npz_2,
                                          D_new_fs=D_npz_2, EII_new_fs=EII_npz_2,
                                          i_frame=i_frame, video_choose=video_choose,
                                          ifAImage=not ifAllVideo, ifShape=True, mode=mode,
                                          path_video=path_video_2, path_Tpose=path_Tpose, path_TposeColor=path_bodyColor,
                                          getImageForColor=getImageForColor_2, path_video_forColor=path_video_2_forColor,
                                          getImageClustered=getImageClustered_2, modeConfig=modeConfig_in_getImageProduced)


            def draw_body_from_tensor(mode, mode_image, i_pose, getImageClustered, path_color_for_drawBody, i,
                                      pInModeCS, mImO_iImage=-1):
                coordinate_2D_tensor_1 = coordinate_2D_tensor[i_pose, :, :, :]
                coordinate_2D_eval = np.float64(sess.run(coordinate_2D_tensor_1))

                radius_2D_tensor_1 = radius_2D_tensor[i_pose, :, :]
                radius_2D_eval = np.float64(sess.run(radius_2D_tensor_1))

                mImO_imagePath = [cf.pathVideo_, mImO_iImage]

                draw_body(mode=mode, mode_image=mode_image,
                          coordinate_2D=coordinate_2D_eval, radius_2D=radius_2D_eval,
                          color=read_bodyColor(path_color_for_drawBody),
                          windowName="times: " + str(i), note="t%d_shape%d_t%d_i%d" % (modeConfig['i_image'], i_pose, i_times, i),
                          imageClustered=getImageClustered,
                          pInModeCS=pInModeCS, mCS_iShape=i,
                          mImO_imagePath=mImO_imagePath)

            time_delta = 0
            if train_times > 0:
                for i in range(train_times * i_frame, train_times * (i_frame + 1) + 1):
                    # log
                    if i % (10) == 0:
                        if if_save_time == True:
                            _, result = sess.run([train_optimizer, merged], feed_dict={
                            B_mu_in_tensor: np.stack([B_mu_npz[0], B_mu_npz_2[0]], axis=0),
                            B_sigma_in_tensor: np.stack([B_sigma_npz[0], B_sigma_npz_2[0]], axis=0),
                            D_in_tensor: np.stack([D_npz[0], D_npz_2[0]], axis=0),
                            EII_tensor: np.stack([EII_npz[0], EII_npz_2[0]], axis=0)},
                                                 options=run_options, run_metadata=run_metadata)
                            writer.add_summary(result, i)
                            writer.add_run_metadata(run_metadata, 'step{}'.format(i))
                        else:
                            _, result = sess.run([train_optimizer, merged], feed_dict={
                            B_mu_in_tensor: np.stack([B_mu_npz[0], B_mu_npz_2[0]], axis=0),
                            B_sigma_in_tensor: np.stack([B_sigma_npz[0], B_sigma_npz_2[0]], axis=0),
                            D_in_tensor: np.stack([D_npz[0], D_npz_2[0]], axis=0),
                            EII_tensor: np.stack([EII_npz[0], EII_npz_2[0]], axis=0)})
                            writer.add_summary(result, i)
                    # print info. 50/16s, 500/160s=3min
                    times_per_print = 500
                    if i % (times_per_print) == 0:
                            if i > 0:
                                if time_delta == 0:
                                    time_delta = datetime.datetime.now() - print_now
                                else:
                                    time_delta = 0.6 * time_delta + 0.4 * (datetime.datetime.now() - print_now)
                            print_now = datetime.datetime.now()

                            print("time: %d_shape    运行次数%d/%d" % (i_times, i, train_times), end="    ")
                            print('当前运行时间：' + str((datetime.datetime.now() - startTime_system)), end="    ")
                            print("时间间隔：" + str(time_delta), end="    ")
                            print("预计剩余时间：" + str(time_delta * (train_times - i) / times_per_print))
                    # save image
                    if (i + 1) % 900 == 0:
                        if 0:
                            color = cf.path_TposeColor
                        else:
                            color = path_bodyColor
                        draw_body_from_tensor("circle", "cluster", 0, getImageClustered_1, color, i, None)
                        draw_body_from_tensor("circle", "cluster", 1, getImageClustered_2, color, i, None)
                    # run
                    sess.run([train_optimizer, merged], feed_dict={
                            B_mu_in_tensor: np.stack([B_mu_npz[0], B_mu_npz_2[0]], axis=0),
                            B_sigma_in_tensor: np.stack([B_sigma_npz[0], B_sigma_npz_2[0]], axis=0),
                            D_in_tensor: np.stack([D_npz[0], D_npz_2[0]], axis=0),
                            EII_tensor: np.stack([EII_npz[0], EII_npz_2[0]], axis=0)})

            # print(sess.run(rl_tensor))
            if if_save_result == True:

                if ifPose == True:
                    axisAngle_3_23_final = np.float64(sess.run(axisAngle_3_23_tensor))
                    root_final = np.float64(sess.run(root_variable))
                    pose_save = np.concatenate((axisAngle_3_23_final, root_final), axis=1)
                    if ifAllVideo == True:
                        SaveAxisAngle(path_save=path_save, axisAngle_3_24=pose_save, filename_save=filename_save + str(i_frame) + ".mat")
                    else:
                        SaveAxisAngle(path_save=path_save, axisAngle_3_24=pose_save, filename_save=filename_save)
                elif ifShape == True:
                    rl_final = np.float64(sess.run(rl_tensor))
                    SaveRL(path_save=path_save, rl=rl_final, filename_save=filename_save)

                    if filename_save_color != "":
                        result_color = np.zeros(shape=[63, 3])
                        coordinate_2D_final = np.float64(sess.run(coordinate_2D_tensor))
                        r_2D_final = np.float64(sess.run(radius_2D_tensor))

                        def get_color(i_shape, num_shape):
                            # 对所有pose，所有view求一个值
                            colorConcrete_allView = []
                            rec = []
                            print("\n当前ball：(sum 63)")
                            for i_body in range(0, 63):

                                if(i_body % 10 == 0 and i_body > 0):
                                    print("\n")
                                print("%3d" % i_body, end="")

                                # get color
                                if 1:
                                    color_view_in_all = []
                                    rec_ball = []
                                    for i_shape in range(i_shape, i_shape + num_shape):
                                        rec_ball_shape = []
                                        for i_view in range(0, 8):

                                            centor = coordinate_2D_final[i_shape][i_view][i_body]
                                            r = r_2D_final[i_shape, i_view, i_body]

                                            # get color
                                            r_coefficient = 0.7
                                            # +1 ?
                                            row_down = max(np.int(np.round(centor[0] - r * r_coefficient)), 1)
                                            row_up = min(np.int(np.round(centor[0] + r * r_coefficient)) + 1, cf.imageWidth)
                                            col_down = max(np.int(np.round(centor[1] - r * r_coefficient)), 1)
                                            col_up = min(np.int(np.round(centor[1] + r * r_coefficient)) + 1, cf.imageHeight)
                                            rec_ball_shape_view=[row_down, row_up, col_down, col_up]
                                            rec_ball_shape.append(rec_ball_shape_view)
                                            for i_row in range(row_down, row_up):
                                                for i_col in range(col_down, col_up):
                                                    color = getImageForColor_1[0][i_view][i_col][i_row]
                                                    color_view_in_all.append(color)
                                        rec_ball.append(rec_ball_shape)
                                    rec.append(rec_ball)

                                    colorConcrete_allView.append(color_view_in_all)
                                    sio.savemat(cf.path_save_direction + "0_colorConcrete_allView.mat",
                                                {'colorConcrete_allView': colorConcrete_allView})

                                # average
                                if 0:
                                    # mean
                                    if 1:
                                        color_all = np.zeros(shape=[3])
                                        for i_color in color_view_in_all:
                                            color_all = color_all + i_color
                                        color_mean = color_all / color_view_in_all.__len__()

                                        print("color_view_in_all: %d" % color_view_in_all.__len__())

                                    # choose
                                    if 1:
                                        # todo: sigma
                                        color_view_choosed = []
                                        color_view_choosed.clear()

                                        # 0.1 > 0.07 > 0.15
                                        color_eta = 0.085
                                        for i_color in color_view_in_all:
                                            if (np.sum(np.power(i_color - color_mean, 2)) < color_eta):
                                                color_view_choosed.append(i_color)
                                        print("color_view_choosed: %d" % color_view_choosed.__len__())

                                    # mean
                                    if 1:
                                        color_all = np.zeros(shape=[3])
                                        for i_color in color_view_choosed:
                                            color_all = color_all + i_color
                                        color_mean = color_all / color_view_choosed.__len__()
                                # median
                                else:
                                    list_RGB_0 = []
                                    list_RGB_1 = []
                                    list_RGB_2 = []
                                    for i_color in color_view_in_all:
                                        list_RGB_0.append(i_color[0])
                                        list_RGB_1.append(i_color[1])
                                        list_RGB_2.append(i_color[2])

                                    list_RGB_0.sort()
                                    list_RGB_1.sort()
                                    list_RGB_2.sort()
                                    position_median = np.int32(np.around(color_view_in_all.__len__() / 2))

                                    color_mean = np.array([list_RGB_0[position_median],
                                                           list_RGB_1[position_median],
                                                           list_RGB_2[position_median]])


                                result_color[i_body, 0:3] = color_mean
                            print(result_color)

                            # draw_body_from_tensor
                            if 1:
                                draw_body_from_tensor("circle_and_square", "origin", 0,
                                                      getImageClustered_1, cf.path_TposeColor, i=0,
                                                      pInModeCS=rec, mImO_iImage=cf.idImage[0])
                                draw_body_from_tensor("circle_and_square", "origin", 1,
                                                      getImageClustered_2, cf.path_TposeColor, i=1,
                                                      pInModeCS=rec, mImO_iImage=cf.idImage[1])

                                sio.savemat(cf.path_save_direction + "0_rec", {'rec': rec})

                            sio.savemat(filename_save_color + ("_pose%d.mat" % (i_shape)), {'color': result_color})

        # pose_final = np.reshape(np.float64(sess.run(pose_3_23_tensor)), (69), order='F')
        # root_final = np.float64(np.squeeze(sess.run(root_variable), axis=1))
        # pose_save = np.concatenate((pose_final, root_final), axis=0)
        #
        # SaveAll(path_save=path_save, pose=pose_save)

                        get_color(i_shape=0, num_shape=1)
                        get_color(i_shape=1, num_shape=1)
                        dlt = 0
            print("frame:" + str(i_frame) + "/" + str(num_frames))
            print('当前运行时间：' + str((datetime.datetime.now() - startTime_system)))


    print('当前运行时间：' + str((datetime.datetime.now() - startTime_system)))

    tf.reset_default_graph()

#
names = locals()
startTime_system = datetime.datetime.now()

# max = 6: trainTimes_shapeInShape_array
num_times = cf.num_times_in_shape

# 优化shape
is_optimize_shape = False
if is_optimize_shape:
    for i_times in range(0, num_times):

        def show_remaining_time(i_times):
            # time
            timesSum_pose = 0
            for i in range(i_times, cf.trainTimes_poseInShape_array.__len__()):
                timesSum_pose = timesSum_pose + cf.trainTimes_poseInShape_array[i]

            timesSum_shape = 0
            for i in range(i_times, cf.trainTimes_shapeInShape_array.__len__()):
                timesSum_shape = timesSum_shape + cf.trainTimes_shapeInShape_array[i]

            timeSum_pose_shape = timesSum_pose * cf.timePer_trainTimes_poseInShape + \
                                 timesSum_shape * cf.timePer_trainTimes_shapeInShape
            print("\nRemaining time of shape: %.1fmin." % (timeSum_pose_shape / 60))


        show_remaining_time(i_times)

        # Pose optimize
        if 1 and i_times >= 0:
            # 2个剪影
            for i_pose in range(1):
                print("\ntimes: %d/%d, pose: %d" % (i_times, num_times, i_pose))
                if i_times == 0:
                    filename_pose = cf.pathVideo_ + "detec_axisAngle_" + str(cf.idImage[i_pose]) + "_3_24_auto.mat"
                    filename_rl = cf.pathfile_rl_init
                else:
                    filename_pose = cf.path_save_direction + \
                                    cf.str_DateTime_save + "_" + str(i_times - 1) + "_pose_" + str(i_pose) + ".mat"
                    filename_rl = \
                        cf.path_save_direction + \
                        cf.str_DateTime_save + "_" + str(i_times - 1) + "_shape" + ".mat"

                filename_save = cf.str_dir_final + cf.str_DateTime_save + "_" + str(i_times) + "_pose_" + \
                                str(i_pose) + ".mat"
                note_out = "_" + cf.str_DateTime_save + "_times" + str(i_times) + "_pose" + str(i_pose)

                path_TposeColor = cf.path_TposeColor
                run(ifPose=True, mode="sil_for_deep",
                    filename_pose=filename_pose, filename_rl=filename_rl, path_TposeColor=path_TposeColor,
                    filename_save=filename_save, note_out=note_out, i_times=i_times, i_poseth=i_pose,
                    frame_start=i_pose*0+i_times)
                    # frame_start=i_pose*10+i_times)

        # Shape optimize
        if 1 and i_times >= 0:
            print("\ntimes: %d/%d, shape" % (i_times, num_times))
            if i_times == 0:
                filename_rl = cf.pathfile_rl_init
            else:
                filename_rl = cf.path_save_direction + cf.str_DateTime_save + "_" + str(i_times - 1)  + "_shape.mat"


            if i_times == num_times - 1:
                filename_save_color=cf.path_save_direction + cf.str_DateTime_save + ("_%d_color.mat" % i_times)
            else:
                filename_save_color = ""

            filename_save = cf.str_dir_final + cf.str_DateTime_save + "_" + str(i_times) + "_shape" + ".mat"
            modeConfig = {'i_image': i_times}
            run2_forShape(ifPose=False, mode="sil_for_deep",
                          filename_pose1=cf.str_DateTime_save + "_" + str(i_times) + "_pose_" + str(0)+ ".mat",
                          filename_pose2=cf.str_DateTime_save + "_" + str(i_times) + "_pose_" + str(0)+ ".mat",
                          filename_rl=filename_rl,
                          filename_save=filename_save, filename_save_color=filename_save_color,
                          note_out="_" + cf.str_DateTime_save + "_shape_" + str(i_times + 1),
                          train_times=cf.trainTimes_shapeInShape_array[i_times],
                          i_times=i_times, modeConfig=modeConfig)

        # Get color
        if 1 and (i_times >= 3):

            print("\ntimes: %d/%d, color" % (i_times, num_times))

            if 1:
                filename_rl = cf.path_save_direction + cf.str_DateTime_save + "_" + str(i_times)  + "_shape"+ ".mat"

                filename_save_color_half = cf.path_save_direction + cf.str_DateTime_save + ("_%d_color" % i_times)
                filename_save = cf.str_dir_final + cf.str_DateTime_save + "_" + str(i_times) + "_shape_dlt" + ".mat"

                modeConfig = {'i_image': i_times}
                run2_forShape(ifPose=False, mode="expect color",
                              filename_pose1=cf.str_DateTime_save + "_" + str(i_times) + "_pose_" + str(0)+ ".mat",
                              filename_pose2=cf.str_DateTime_save + "_" + str(i_times) + "_pose_" + str(0)+ ".mat",
                              filename_rl=filename_rl,
                              filename_save=filename_save,
                              note_out="_" + cf.str_DateTime_save + "_shape_color_" + str(i_times),
                              filename_save_color=filename_save_color_half,
                              train_times=0,
                              i_times=i_times, modeConfig=modeConfig)
            # Draw body
            def draw_body_in_main(i_pose):
                filename_pose = cf.path_save_direction + \
                                cf.str_DateTime_save + "_" + str(i_times) + "_pose_" + str(i_pose) + ".mat"

                filename_rl = cf.path_save_direction + \
                              cf.str_DateTime_save + "_" + str(i_times) + "_shape.mat"

                filename_save = cf.str_dir_final + cf.str_DateTime_save + "_pose_"
                note_out = "_" + cf.str_DateTime_save + "_pose"

                if 1:
                    path_TposeColor_optimized = cf.path_save_direction + cf.str_DateTime_save \
                                                + ("_%d_color_pose%d" % (i_times, i_pose))
                else:
                    path_TposeColor_optimized = cf.path_save_direction + cf.str_DateTime_save \
                                                + ("_%d_color" % (i_times))

                # pathFile_TposeColor_optimized = path_TposeColor_optimized + ".mat"

                run(ifPose=True, mode="expect color",
                    filename_pose=filename_pose, filename_rl=filename_rl, path_TposeColor=path_TposeColor_optimized,
                    filename_save=filename_save, note_out=note_out,
                    ifAllVideo=True, frame_start=cf.idImage[i_pose], num_frames=1, i_times=i_times, i_poseth=i_pose)

            draw_body_in_main(i_pose=0)
            if 0:
                draw_body_in_main(i_pose=1)


# 优化pose
if 1:
    i_image = 2000

    if is_optimize_shape == True:
        filename_pose = cf.path_save_direction + \
                        cf.str_DateTime_save + "_" + str(num_times - 1) + "_pose_" + str(0) + ".mat"
        filename_rl = cf.path_save_direction + \
                      cf.str_DateTime_save + "_" + str(num_times - 1) + "_shape.mat"
        path_TposeColor_optimized = cf.path_save_direction + cf.str_DateTime_save + ("_%d_color_pose0" % (num_times - 1))
    else:
        str_DateTime_save_tmp = "181128"
        # str_DateTime_save_tmp = cf.str_DateTime_save_last
        filename_pose = "%s%s_pose_%d.mat" % (cf.path_save_direction, str_DateTime_save_tmp, i_image - 1)
        # todo: choose
        # filename_pose = "%s%s_pose_%d.mat" % (cf.path_save_direction, "181211", i_image - 1)
        filename_rl = "%s%s_%d_shape.mat" % (cf.path_save_direction, str_DateTime_save_tmp, num_times - 1)
        path_TposeColor_optimized = "%s%s_%d_color_pose0" % (cf.path_save_direction, str_DateTime_save_tmp, num_times - 1)


    pathFile_TposeColor_optimized = path_TposeColor_optimized + ".mat"

    filename_save = cf.str_dir_final + cf.str_DateTime_save + "_pose_"
    note_out = "_" + cf.str_DateTime_save + "_pose"

    max_of_cluster = 0
    # frame_start=cf.idImage[0]
    frame_start = i_image
    frame_end = 3600
    run(ifPose=True,mode="pose_in_all_video",
        filename_pose=filename_pose, filename_rl=filename_rl, path_TposeColor=path_TposeColor_optimized,
        filename_save=filename_save, note_out=note_out,
        ifAllVideo=True, frame_start=frame_start, num_frames=frame_end-frame_start, max_of_cluster=max_of_cluster)


