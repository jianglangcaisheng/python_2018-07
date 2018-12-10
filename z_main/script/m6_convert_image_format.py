
import os
import shutil
import cv2
import utility

def extract_2bmpAnd2jpg(file_root_dir, image_goal_list):
    assert False, "aborted"
    import cv2
    for i_image in image_goal_list:
        for i_cam in range(0, 8):
            file_goal = "%d-%d.bmp" % (i_image, i_cam)
            file_dir = file_root_dir + "%d" % i_cam
            file_col_dir = file_root_dir + "/col/"
            if not os.path.exists(file_col_dir):
                os.mkdir(file_col_dir)

            for root, dirs, files in os.walk(file_dir):
                for i_file in range(files.__len__()):
                    if files[i_file] == file_goal:
                        shutil.copy(file_dir + "/" + files[i_file], file_col_dir + files[i_file])

                        image = cv2.imread(file_col_dir + files[i_file])
                        cv2.imwrite(file_col_dir + "%d-%d.jpg" % (i_image, i_cam), img=image)



path_image_list = []
pathGoal_image_list = []
for i_video in range(5, 7):
    pathRoot_image = "D:/0_image_1811_1_cali/"
    path_image_list.append(pathRoot_image + "K1_RT1_D%d/" % i_video)
    pathGoal_image_list.append(pathRoot_image+ "K1_RT1_D%d_jpg/" % i_video)

for i_path_image in range(path_image_list.__len__()):

    def bmp2jpg(path_image, pathGoal_image):
        utility.mkdir(pathGoal_image)
        for i_view in range(8):
            path_image_view = "%s%d/" % (path_image, i_view)
            pathGoal_image_view = "%s%d/" % (pathGoal_image, i_view)
            utility.mkdir(pathGoal_image_view)
            for root, dirs, files in os.walk(path_image_view):
                for i_file in range(files.__len__()):
                    pass
                    image = cv2.imread(path_image_view + files[i_file])
                    pathFileSave = pathGoal_image_view + files[i_file][0:3] + ".jpg"
                    cv2.imwrite(pathFileSave, img=image)
                    print("Finish %s%s" % (pathFileSave))

    bmp2jpg(path_image_list[i_path_image], pathGoal_image_list[i_path_image])