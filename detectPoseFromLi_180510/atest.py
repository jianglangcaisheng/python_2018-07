import tensorflow as tf
import numpy as np
import cv2
import datetime
import scipy.io as sio

from detectPoseFromLi_180510.hg import Model
import z_main.utility as utility


# ---------------------------------------------------------------------------------------------
def getPreds(hm, useThresh=False, thresh=0.1):
    # hm: numpy array, H*W*C or N*H*W*C
    if len(hm.shape) == 3:
        return getPreds(np.array([hm]), useThresh, thresh)[0]
    preds = np.zeros([hm.shape[0], hm.shape[3], 2], dtype=np.int32)
    for imgidx in range(hm.shape[0]):
        for jidx in range(hm.shape[3]):
            preds[imgidx, jidx, :] = getMaxPos(hm[imgidx, :, :, jidx], useThresh, thresh)
    return preds


def getMaxPos(hm, useThresh, thresh):
    # hm: numpy array, H*W
    if useThresh and np.max(hm) < thresh:
        return np.array([-1, -1], dtype=np.int32)
    else:
        pos = np.argmax(hm)
        posy = int(pos / hm.shape[1])
        posx = pos - posy * hm.shape[1]
        return np.array([posx, posy], dtype=np.int32)


# ---------------------------------------------------------------------------------------------
def get_pose_by_network(path_image):
    if_small_image = False

    begin = datetime.datetime.now()
    last_time = begin

    # path_image = "D:/0_image_1811_1_cali\K1_RT1_D3/extract_body/"
    pathFileSave = path_image
    utility.mkdir(pathFileSave)
    pathSave_image = pathFileSave + "image(large)/"
    utility.mkdir(pathSave_image)

    images = np.zeros([8, 256, 256, 3], dtype=np.float32)
    # images_origin = np.zeros([8, 1536, 2048, 3], dtype=np.float32)
    images_origin_list = [0, 1, 2, 3, 4, 5, 6, 7]
    images_origin_size_list = [0, 1, 2, 3, 4, 5, 6, 7]

    print("Before Model()")
    model = Model()
    print("After Model()")
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        time_run = datetime.datetime.now()

        image_end = 151
        for i_image in range(150, image_end):

            for i_view in range(0, 8):
                pathFile_image = path_image + "%d-%d.jpg" % (i_image, i_view)
                img_origin = cv2.imread(pathFile_image)
                if img_origin is None:
                    print(pathFile_image)
                    assert img_origin, "img_origin is None"

                images_origin_list[i_view] = img_origin
                images_origin_size_list[i_view] = [img_origin.shape[0], img_origin.shape[1]]

                if (img_origin.shape[0] > 256 or img_origin.shape[1] > 256):
                    img = cv2.resize(img_origin, (256, 256))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                images[i_view] = img

            out = sess.run(model.output, feed_dict={model.inputs: images})

            preds = getPreds(out[3]) * 4

            # dimension processing
            if 1:
                preds_fixed = preds.copy()
                for i_view in range(8):
                    preds_fixed[i_view, :, 0] = preds_fixed[i_view, :, 0] * images_origin_size_list[i_view][1] / 256
                    preds_fixed[i_view, :, 1] = preds_fixed[i_view, :, 1] * images_origin_size_list[i_view][0] / 256

                preds_fixed_dim = np.swapaxes(preds_fixed, 0, 1)
                preds_fixed_dim = np.swapaxes(preds_fixed_dim, 1, 2)

                co_output = np.zeros(shape=[15, 2, 8])
                co_output[0:8, :, :] = preds_fixed_dim[0:8, :, :]
                co_output[8:15, :, :] = preds_fixed_dim[9:16, :, :]


                filename_save = "coByNet_%d.mat" % (i_image)
                sio.savemat(pathFileSave+ filename_save, {'co': co_output})


            def draw_line(img, preds, i_view):
                pt1s = [(preds[i_view, 0, 0], preds[i_view, 0, 1]),
                        (preds[i_view, 1, 0], preds[i_view, 1, 1]),
                        (preds[i_view, 2, 0], preds[i_view, 2, 1]),
                        (preds[i_view, 6, 0], preds[i_view, 6, 1]),
                        (preds[i_view, 3, 0], preds[i_view, 3, 1]),
                        (preds[i_view, 4, 0], preds[i_view, 4, 1]),
                        (preds[i_view, 6, 0], preds[i_view, 6, 1]),
                        (preds[i_view, 7, 0], preds[i_view, 7, 1]),
                        (preds[i_view, 8, 0], preds[i_view, 8, 1]),
                        (preds[i_view, 10, 0], preds[i_view, 10, 1]),
                        (preds[i_view, 11, 0], preds[i_view, 11, 1]),
                        (preds[i_view, 12, 0], preds[i_view, 12, 1]),
                        (preds[i_view, 7, 0], preds[i_view, 7, 1]),
                        (preds[i_view, 13, 0], preds[i_view, 13, 1]),
                        (preds[i_view, 14, 0], preds[i_view, 14, 1])
                        ]
                pt2s = [(preds[i_view, 1, 0], preds[i_view, 1, 1]),
                        (preds[i_view, 2, 0], preds[i_view, 2, 1]),
                        (preds[i_view, 6, 0], preds[i_view, 6, 1]),
                        (preds[i_view, 3, 0], preds[i_view, 3, 1]),
                        (preds[i_view, 4, 0], preds[i_view, 4, 1]),
                        (preds[i_view, 5, 0], preds[i_view, 5, 1]),
                        (preds[i_view, 7, 0], preds[i_view, 7, 1]),
                        (preds[i_view, 8, 0], preds[i_view, 8, 1]),
                        (preds[i_view, 9, 0], preds[i_view, 9, 1]),
                        (preds[i_view, 11, 0], preds[i_view, 11, 1]),
                        (preds[i_view, 12, 0], preds[i_view, 12, 1]),
                        (preds[i_view, 7, 0], preds[i_view, 7, 1]),
                        (preds[i_view, 13, 0], preds[i_view, 13, 1]),
                        (preds[i_view, 14, 0], preds[i_view, 14, 1]),
                        (preds[i_view, 15, 0], preds[i_view, 15, 1])
                        ]
                for i_point in range(pt1s.__len__()):
                    cv2.line(img=img, pt1=pt1s[i_point], pt2=pt2s[i_point], color=(255, 0, 0))
                return img

            for i_joint in range(0, 1):
                for i_view in range(8):

                    if if_small_image:
                        img = images[i_view] * 255
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    else:
                        img = images_origin_list[i_view]

                    if if_small_image:
                        for j in range(0, 16):
                            cv2.circle(img, (preds[i_view, j, 0], preds[i_view, j, 1]), 1, (0, 255, 0), 3)

                        img = draw_line(img, preds, i_view)

                        img = cv2.resize(img, (2048, 1536))
                    else:
                        for j in range(0, 16):
                            cv2.circle(img, (preds_fixed[i_view, j, 0], preds_fixed[i_view, j, 1]), 1 * 6, (0, 255, 0), 3 * 6)
                        img = draw_line(img, preds_fixed, i_view)


                    if 1:
                        cv2.imwrite(pathSave_image + ("000_after_result_%d-%d.bmp" % (i_image, i_view)), img)
                        # cv2.imwrite(pathSave_image + ("result_%d-%d.bmp" % (i_image, i_view)), img)
                    else:
                        cv2.imwrite(pathSave_image + ("result_j%d_%d-%d.bmp" % (i_joint, i_image, i_view)), img)




            # time manager
            if 0:
                time_run_last = time_run
                time_run = datetime.datetime.now()
                time_run_delta = time_run - time_run_last
                print('finish: %d, 总运行时间：%s， 时间间隔：%s，预计完成时间：%s' %
                      (i_image,
                       str(datetime.datetime.now() - begin),
                       str(time_run_delta),
                       str(time_run_delta * (image_end - i_image))))
            elif 1:
                last_time = utility.timeManager(begin_time=begin, last_time=last_time, id=i_image, remainings=(image_end - i_image))
