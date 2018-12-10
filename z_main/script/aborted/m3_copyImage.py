

import os, sys
import datetime
import shutil


def extract_image_K(file_root_dir):

    for i_view in range(8):
        print("Processing: %d" % i_view)
        file_dir = file_root_dir + ("origin/%d/" % i_view)

        file_dir_dst = file_root_dir + ("/%d/" % i_view)
        if not os.path.exists(file_dir_dst):
            os.mkdir(file_dir_dst)

        for root, dirs, files in os.walk(file_dir):
            for i_image in range((i_view % 4) * 25, ((i_view % 4) + 1) * 25):
                file_goal = "%d-%d.bmp" % (i_image, i_view)
                for i_file in range(files.__len__()):
                    if files[i_file] == file_goal:
                        shutil.copy(file_dir + files[i_file], file_dir_dst + files[i_file])


def copy_image_ToFrom(path_src, path_dst, dirs):
    print("\nProcessing: %s" % path_src)
    time_begin = datetime.datetime.now()
    for i_view in dirs:
        print("Processing: %d" % i_view)
        shutil.copytree(path_src + str(i_view), path_dst + str(i_view), False)
    print("Elapsed time: %s" % str(datetime.datetime.now() - time_begin))


if __name__ == "__main__":
    # once, half: 0:49 + 2:26 + 10:08 = 13:23
    # once, half: 0:49 + 1:53 + 5:44 = 8:26

    list_0123 = [0, 1, 2, 3]
    list_4567 = [4, 5, 6, 7]

    # PC1, PC2. first 1 then 2.
    PC = "PC1"
    if PC == "PC1":
        list1 = list_0123
        list2 = list_4567
        pathName_mobileDisk = "U"
        direction_disk2Mobile = False
        direction_Mobile2disk = not direction_disk2Mobile
    else:
        list1 = list_4567
        list2 = list_0123
        pathName_mobileDisk = "I"
        direction_disk2Mobile = True
        direction_Mobile2disk = True

    param_K = 2
    param_RT = 1

    note = "1810_9"
    path_disks = []
    path_mobileDisks = []

    if 0:
        path_disks.append(r"D:\0_image_%s_cali\0_image_%s_cali_K%d/" % (note, note, param_K))
        path_mobileDisks.append(r"%s:\0_image_%s_cali\0_image_%s_cali_K%d/" %
        (pathName_mobileDisk, note, note, param_K))

        path_disks.append(r"D:\0_image_%s_cali\0_image_%s_cali_K%d_RT1/" % (note, note, param_K))
        path_mobileDisks.append(r"%s:\0_image_%s_cali\0_image_%s_cali_K%d_RT1/" %
        (pathName_mobileDisk, note, note, param_K))

    for param_D in range(12, 17):
        path_disks.append(
            r"D:\0_image_%s_cali\0_image_%s_cali_K%d_RT1_D%d/" % (note, note, param_K, param_D))
        path_mobileDisks.append(
            r"%s:\0_image_%s_cali\0_image_%s_cali_K%d_RT1_D%d/" %
            (pathName_mobileDisk, note, note, param_K, param_D))



    # mobile 2 disk
    if direction_Mobile2disk:
        for i in range(0, path_disks.__len__()):
            copy_image_ToFrom(path_mobileDisks[i], path_disks[i], list2)

    # disk 2 mobile
    if direction_disk2Mobile:
        for i in range(0, path_disks.__len__()):
            copy_image_ToFrom(path_disks[i], path_mobileDisks[i], list1)
