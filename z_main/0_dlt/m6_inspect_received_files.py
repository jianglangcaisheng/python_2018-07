
import os
import cv2

import config as cf

def inspect_image():

    for i_frame in range(150, 600):
        for i_view in range(4, 8):
            # txt
            if 0:
                length = {"B": 0, "C": 0}
                for i_file in ["B", "C"]:
                    pathFile = "%sv%d/image%d_%s.txt" % (cf.pathVideo_, i_view, i_frame, i_file)
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
                pathFile = "%s%d/%d-%d.bmp" % (cf.pathVideo_, i_view, i_frame, i_view)
                if not os.path.exists(pathFile):
                    print("File not exists: %s" % pathFile)
                else:
                    image = cv2.imread(pathFile)
                    if image is None:
                        print("File error opening: %s" % pathFile)

    print("Finish inspecting.")

if __name__ == "__main__":
    assert False, "aborted"