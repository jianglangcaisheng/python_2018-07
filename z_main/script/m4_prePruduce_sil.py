
import numpy as np
import cv2
import function_0505_2_crossPlatform.module_Lab as mLab

path_image = r"D:\0_image_1810_9_cali\0_image_1810_9_cali_K2_RT1_D3/"
path_imageBlank = r"D:\0_image_1810_9_cali\0_image_1810_9_cali_K2_RT1_D6/"

width = 2048
height = 1536

width_imshow = int(width / 2)
height_imshow = int(height / 2)

for i_view in range(0, 1):
    for i_image in range(150, 151):
        pathName_image = "%s%d/%d-%d.bmp" % (path_image, i_view, i_image, i_view)
        pathName_imageBlank = "%s%d/%d-%d.bmp" % (path_imageBlank, i_view, i_image, i_view)
        image_flag = 1

        image = cv2.imread(pathName_image, image_flag)
        image = cv2.resize(image, (width_imshow, height_imshow))
        imageBlank = cv2.imread(pathName_imageBlank, image_flag)
        imageBlank = cv2.resize(imageBlank, (width_imshow, height_imshow))

        def preProcess_image(image, windowName):
            image_result = image.copy()
            if 0:
                image_result[:, :, 1] = 0
            elif 0:
                image_result = image_result[:, :, 1]
            elif 0:
                image_result = mLab.BGR2LAB(image_result)
                image_result[:, :, 0] = 70
                image_result = mLab.LAB2BGR(image_result)

            if 1:
                cv2.namedWindow(windowName)
                cv2.imshow(windowName, image_result)

            return image_result


        image = preProcess_image(image, "Image")
        imageBlank = preProcess_image(imageBlank, "imageBlank")


        def filter_image(image, windowName):
            if 0:
                image_result = cv2.medianBlur(image, 9)
            elif 1:
                image_result = cv2.blur(image, (5, 5))

            if 0:
                cv2.namedWindow(windowName)
                cv2.imshow(windowName, image_result)
            return image_result

        image_filtered = filter_image(image, "image_filtered")
        imageBlank_filtered = filter_image(imageBlank, "imageBlank_filtered")

        imageDelta = abs(np.float64(image) - np.float64(imageBlank_filtered))
        imageDelta_amplified = ((imageDelta > 0) + 0) * 100
        imageDelta_uint8 = np.uint8(imageDelta_amplified)

        if 0:
            cv2.namedWindow("imageDelta_v%d" % i_view)
            cv2.imshow("imageDelta_v%d" % i_view, imageDelta_uint8)

cv2.waitKey(0)