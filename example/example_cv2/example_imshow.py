import os
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
import imageio
import skimage

N = 0
imgPath = "J:\\0 SOG\SOG.png"
savePath = "J:\\0 SOG"


if not os.path.exists(savePath):
    os.mkdir(savePath)

Img = cv2.imread(imgPath)
size = Img.shape
width = size[0]
height = size[1]



# image = skimage.img_as_float(im).astype(np.float64)
# fig = pylab.figure()
# fig.suptitle('image #{}'.format(1), fontsize=20)
# pylab.imshow(Img)  # image array 均可
# pylab.show()

# cv2.namedWindow("Image")
# cv2.imshow("Image", Img)

NewImg = Img

# RGB2Gray
GrayImg = cv2.cvtColor(NewImg, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow("Image")
# cv2.imshow("Image", GrayImg)
# cv2.waitKey (0)


# RGB2HSV
HSVImg = cv2.cvtColor(NewImg, cv2.COLOR_BGR2HSV)
# cv2.namedWindow("Image")
# cv2.imshow("Image", HSVImg)
# cv2.waitKey (0)

# RGB2lab
LABImg = cv2.cvtColor(NewImg, cv2.COLOR_BGR2LAB)
cv2.namedWindow("LABImg")
cv2.imshow("LABImg", LABImg)

BGRImg = cv2.cvtColor(LABImg, cv2.COLOR_LAB2BGR)
cv2.namedWindow("RGBImg")
cv2.imshow("RGBImg", BGRImg)
cv2.waitKey (0)
