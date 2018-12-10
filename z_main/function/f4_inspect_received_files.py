
import os
import cv2

import config as cf
import utility

def inspect_image(range_frame, range_view, cf_for_m5):

    for i_frame in range_frame:
        for i_view in range_view:
            # txt
            if 0:
                length = {"B": 0, "C": 0}
                for i_file in ["B", "C"]:
                    pathFile = "%sv%d_bmp/image%d_%s.txt" % (cf_for_m5.path_image, i_view, i_frame, i_file)
                    if not os.path.exists(pathFile):
                        print("File not exists: %s" % pathFile)

                    file = open(pathFile, 'r')
                    length[i_file] = len(file.readlines())
                    file.close()
                if length["B"] != length["C"]:
                    print("frame: %d, view: %d" % (i_frame, i_view))
                    print(length)
            # image
            if 1:
                pathFile = "%s%d_bmp/%d-%d.bmp" % (cf_for_m5.path_image, i_view, i_frame, i_view)
                if not os.path.exists(pathFile):
                    print("File not exists: %s" % pathFile)
                else:
                    image = cv2.imread(pathFile)
                    if image is None:
                        print("File error opening: %s" % pathFile)

    print("Finish inspecting.")


# def inspect_image_cluster(range_frame, range_view):
#
#     for i_frame in range_frame:
#         for i_view in range_view:
#             length = {"B": 0, "C": 0}
#             for i_file in ["B", "C"]:
#                 pathFile = "%sv%d/image%d_%s.txt" % (cf_for_m5.path_image, i_view, i_frame, i_file)
#                 if not os.path.exists(pathFile):
#                     utility.str2red("File not exists: %s" % pathFile)
#
#                 file = open(pathFile, 'r')
#                 length[i_file] = len(file.readlines())
#                 file.close()
#             if length["B"] != length["C"]:
#                 utility.str2red("Error length: frame: %d, view: %d" % (i_frame, i_view))
#                 print(length)
