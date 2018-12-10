
# import openpyxl as xls
import os, sys
import shutil


def rename(file_dir_half):
    for i_cam in range(0, 8):
        file_dir = file_dir_half + "%d" % i_cam

        for root, dirs, files in os.walk(file_dir):
            print("Processing: %d, 共计%d个文件。" % (i_cam, files.__len__()))
            for i_file in range(files.__len__()):
                os.renames(file_dir + "\\" + files[i_file], file_dir + "\\" + "%d--%d.bmp" % (i_file, i_cam))

        for root, dirs, files in os.walk(file_dir):
            for i_file in range(files.__len__()):
                os.renames(file_dir + "\\" + files[i_file], file_dir + "\\" + "%d-%d.bmp" % (i_file, i_cam))


def extract_2bmp(file_root_dir, image_goal_list):
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


def extract_2bmpAnd2jpg(file_root_dir, image_goal_list):
    import cv2
    for i_image in image_goal_list:
        for i_cam in range(0, 8):
            file_goal = "%d-%d.bmp" % (i_image, i_cam)
            file_dir = file_root_dir + "%d_bmp" % i_cam
            file_col_dir = file_root_dir + "/col/"
            if not os.path.exists(file_col_dir):
                os.mkdir(file_col_dir)

            for root, dirs, files in os.walk(file_dir):
                for i_file in range(files.__len__()):
                    if files[i_file] == file_goal:
                        shutil.copy(file_dir + "/" + files[i_file], file_col_dir + files[i_file])

                        image = cv2.imread(file_col_dir + files[i_file])
                        cv2.imwrite(file_col_dir + "%d-%d.jpg" % (i_image, i_cam), img=image)


def extract_image_K(file_root_dir):

    num_image = 20

    for i_view in range(8):
        print("Processing: %d" % i_view)
        file_dir = file_root_dir + ("origin/%d/" % i_view)

        file_dir_dst = file_root_dir + ("%d/" % i_view)
        if not os.path.exists(file_dir_dst):
            os.mkdir(file_dir_dst)

        for root, dirs, files in os.walk(file_dir):
            for i_image in range((i_view % 4) * num_image + 1, ((i_view % 4) + 1) * num_image + 1):
                file_goal = "%d-%d.bmp" % (i_image, i_view)
                for i_file in range(files.__len__()):
                    if files[i_file] == file_goal:
                        shutil.copy(file_dir + files[i_file], file_dir_dst + files[i_file])


def cut_sil(file_root_dir, image_goal_list):
    for i_image in image_goal_list:
        number_of_moving = 0
        for i_cam in range(0, 8):
            if 1:
                file_goal = "%d-%d_sil.jpg" % (i_image, i_cam)
            else:
                file_goal = "%d-%d.jpg" % (i_image, i_cam)
            file_dir = file_root_dir + "col/"

            file_dst = "%d-%d.jpg" % (i_image, i_cam)
            file_dst_dir = file_root_dir + "sil/"
            if not os.path.exists(file_dst_dir):
                os.mkdir(file_dst_dir)

            # number_of_moving = 0
            for root, dirs, files in os.walk(file_dir):
                for i_file in range(files.__len__()):
                    if files[i_file] == file_goal:
                        shutil.move(file_dir + "/" + files[i_file], file_dst_dir + file_dst)
                        number_of_moving = number_of_moving + 1
                        print("number_of_moving: %d" % number_of_moving)

if __name__ == "__main__":

    file_root_dir = r"D:\0_image_1811_2_cali/"
    param_K = 1
    param_RT = 1
    param_D = 1
    path_K = file_root_dir + ("K%d/" % param_K)
    path_KR = file_root_dir + ("K%d_RT%d/" % (param_K, param_RT))
    path_KRD = file_root_dir + ("K%d_RT%d_D%d/" % (param_K, param_RT, param_D))
    image_goal_list = [150, 150]

    # create dir origin
    if 0:
        extract_image_K(file_root_dir = path_K)

    # after picking up 15+
    if 1:
        rename(file_dir_half=path_K)

    exit(666)

    # extract bmp
    if 0:
        extract_2bmpAnd2jpg(file_root_dir=path_KRD, image_goal_list=image_goal_list)
        # extract_2bmp(file_root_dir=path_KRD, image_goal_list=image_goal_list)

    # cut sil. recommand copy before cutting
    if 0:
        cut_sil(file_root_dir=path_KRD, image_goal_list=image_goal_list)