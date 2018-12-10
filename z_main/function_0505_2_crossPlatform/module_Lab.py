
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
import imageio
import skimage


def BGR2RGB(image):
    R = image[:, :, 2]
    G = image[:, :, 1]
    B = image[:, :, 0]
    return np.stack([R, G, B], 2)


def RGB2BGR(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    return np.stack([B, G, R], 2)


def BGR2LAB(image):
    """requested: uint double // out: double"""
    R = image[:, :, 2]
    G = image[:, :, 1]
    B = image[:, :, 0]
    if(np.max(R)>1):
        R = R / 255.
        G = G / 255.
        B = B / 255.

    M = np.size(R, 0)
    N = np.size(R, 1)
    s = M * N
    T = 0.008856

    RGB = np.transpose(np.stack((np.reshape(R, s), np.reshape(G, s), np.reshape(B, s)), axis=1))

    MAT = np.array([[0.412453, 0.357580, 0.180423],
                    [0.212671, 0.715160, 0.072169],
                    [0.019334, 0.119193, 0.950227]])

    XYZ = np.matmul(MAT, RGB)
    X = XYZ[0, :] / 0.950456
    Y = XYZ[1, :]
    Z = XYZ[2, :] / 1.088754

    XT = X > T
    YT = Y > T
    ZT = Z > T

    fX = XT * np.power(X, 1./3.) + np.logical_not(XT) * (7.787 * X + 16./116.)
    Y3 = np.power(Y, 1./3.)
    fY = YT * Y3 + np.logical_not(YT) * (7.787 * Y + 16./116.)
    L = YT * (116. * Y3 - 16.) + np.logical_not(YT) * (903.3 * Y)

    fZ = ZT * np.power(Z, 1./3.) + np.logical_not(ZT) * (7.787 * Z + 16./116.)

    a = 500. * (fX - fY)
    b = 200. * (fY - fZ)

    L = np.reshape(L, [M, N])
    a = np.reshape(a, [M, N])
    b = np.reshape(b, [M, N])

    image_Lab = np.stack((L, a, b), 2)
    return image_Lab

# same
def RGB2LAB(image):
    """requested: uint double // out: double"""
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    if(np.max(R)>1):
        R = R / 255.
        G = G / 255.
        B = B / 255.

    M = np.size(R, 0)
    N = np.size(R, 1)
    s = M * N
    T = 0.008856

    RGB = np.transpose(np.stack((np.reshape(R, s), np.reshape(G, s), np.reshape(B, s)), axis=1))

    MAT = np.array([[0.412453, 0.357580, 0.180423],
                    [0.212671, 0.715160, 0.072169],
                    [0.019334, 0.119193, 0.950227]])

    XYZ = np.matmul(MAT, RGB)
    X = XYZ[0, :] / 0.950456
    Y = XYZ[1, :]
    Z = XYZ[2, :] / 1.088754

    XT = X > T
    YT = Y > T
    ZT = Z > T

    fX = XT * np.power(X, 1./3.) + np.logical_not(XT) * (7.787 * X + 16./116.)
    Y3 = np.power(Y, 1./3.)
    fY = YT * Y3 + np.logical_not(YT) * (7.787 * Y + 16./116.)
    L = YT * (116. * Y3 - 16.) + np.logical_not(YT) * (903.3 * Y)

    fZ = ZT * np.power(Z, 1./3.) + np.logical_not(ZT) * (7.787 * Z + 16./116.)

    a = 500. * (fX - fY)
    b = 200. * (fY - fZ)

    L = np.reshape(L, [M, N])
    a = np.reshape(a, [M, N])
    b = np.reshape(b, [M, N])

    image_Lab = np.stack((L, a, b), 2)
    return image_Lab


def LAB2RGB(image_Lab):
    """requested: double // out: uint8"""
    L = image_Lab[:, :, 0]
    a = image_Lab[:, :, 1]
    b = image_Lab[:, :, 2]

    T1 = 0.008856
    T2 = 0.206893

    [M, N] = np.shape(L)
    s = M * N
    L = np.reshape(L, s)
    a = np.reshape(a, s)
    b = np.reshape(b, s)

    fY = np.power((L + 16) / 116, 3.)
    YT = fY > T1
    fY = np.logical_not(YT) * (L / 903.3) + YT * fY
    Y = fY

    fY = YT * np.power(fY, 1./3.) + np.logical_not(YT) * (7.787 * fY + 16./116.)

    fX = a / 500. + fY
    XT = fX > T2
    X = (XT * np.power(fX, 3.)) + np.logical_not(XT) * ((fX - 16./116.) / 7.787)

    fZ = fY - b / 200.
    ZT = fZ > T2

    Z = (ZT * np.power(fZ, 3.) + np.logical_not(ZT) * ((fZ - 16./116.) / 7.787))

    X = X * 0.950456
    Z = Z * 1.088754

    MAT = np.array([[3.240479, -1.537150, -0.498535],
                    [-0.969256, 1.875992, 0.041556],
                    [0.055648, -0.204043, 1.057311]])

    RGB_raw = np.matmul(MAT, np.transpose(np.stack((X, Y, Z), axis=1)))
    dlt = np.max(RGB_raw)
    RGB = np.maximum(np.minimum(RGB_raw, 1.), 0.)

    image_RGB_R = np.reshape(RGB[0, :], [M, N]) * 255.
    image_RGB_G = np.reshape(RGB[1, :], [M, N]) * 255.
    image_RGB_B = np.reshape(RGB[2, :], [M, N]) * 255.

    image_RGB = np.uint8(np.round(np.stack((image_RGB_R, image_RGB_G, image_RGB_B), 2)))
    return image_RGB


def LAB2BGR(image_Lab):
    return RGB2BGR(LAB2RGB(image_Lab))


if __name__ == "__main__":
    imgPath = r"U:\0 image_1809_6_cali_K1_RT1_D1\1\200-1.bmp"

    Img = cv2.imread(imgPath)
    size = Img.shape
    width = size[0]
    height = size[1]
    # imshow
    if 0:
        cv2.namedWindow("Img")
        cv2.imshow("Img", Img)

    # BGR
    if 0:
        image_B = Img[:, :, 0]
        image_G = Img[:, :, 1]
        image_R = Img[:, :, 2]
        cv2.namedWindow("image_B")
        cv2.imshow("image_B", image_B)
        cv2.namedWindow("image_G")
        cv2.imshow("image_G", image_G)
        cv2.namedWindow("image_R")
        cv2.imshow("image_R", image_R)

    for L in range(50, 90 + 1, 5):

        # myLab
        if 1:
            LABImg = BGR2LAB(Img)
            LABImg[:, :, 0] = L
            LABImg = LAB2RGB(LABImg)
            LABImg = RGB2BGR(LABImg)
            cv2.namedWindow("LABImg_%d" % L)
            cv2.imshow("LABImg_%d" % L, LABImg)

        # cv2.Lab
        if 0:
            LABImg_CV = cv2.cvtColor(Img, cv2.COLOR_BGR2LAB)
            LABImg_CV[:, :, 0] = L
            LABImg_CV = cv2.cvtColor(LABImg_CV, cv2.COLOR_LAB2BGR)
            cv2.namedWindow("LABImg_CV")
            cv2.imshow("LABImg_CV", LABImg_CV)
            # image_L = LABImg[:, :, 0]
            # image_a = LABImg[:, :, 1]
            # image_b = LABImg[:, :, 2]


        print("imshow L = %d" % L)
    cv2.waitKey (0)

