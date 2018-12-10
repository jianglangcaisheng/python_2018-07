import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
import imageio
import skimage

# save
if 0:
    savePath = "J:\\0 SOG"
    if not os.path.exists(savePath):
        os.mkdir(savePath)

# imshow
if 1:
    imgPath = "J:\\0 SOG\SOG.png"
    Img = cv2.imread(imgPath)

    green = (0, 255, 0)
    cv2.line(Img, (0, 0), (300, 300), green)

    cv2.namedWindow("Image")
    cv2.imshow("Image", Img)
    cv2.waitKey (0)

if 0:
    NewImg = Img
    GrayImg = cv2.cvtColor(NewImg, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("Image")
    cv2.imshow("Image", GrayImg)
    cv2.waitKey (0)

    HSVImg = cv2.cvtColor(NewImg, cv2.COLOR_BGR2HSV)
    cv2.namedWindow("Image")
    cv2.imshow("Image", HSVImg)
    cv2.waitKey (0)

    LABImg = cv2.cvtColor(NewImg, cv2.COLOR_BGR2LAB)
    cv2.namedWindow("LABImg")
    cv2.imshow("LABImg", LABImg)

    BGRImg = cv2.cvtColor(LABImg, cv2.COLOR_LAB2BGR)
    cv2.namedWindow("RGBImg")
    cv2.imshow("RGBImg", BGRImg)
    cv2.waitKey (0)
