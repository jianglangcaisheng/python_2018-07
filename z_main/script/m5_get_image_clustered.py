
import datetime
import os
import numpy as np

import z_main.config as cf
from z_main.function.image_produce import *
from z_main.function.e3_file_send import *
import function.f4_inspect_received_files

class CF_for_m5:
    def __init__(self, is_inspect, is_clusterd, is_send_image):
        self.is_inspect = is_inspect
        self.is_clusterd = is_clusterd
        self.is_send_image = is_send_image
        self.range_resend = None
        self.path_image = None
        self.num_frames = None


class CF_for_m5_2:
    def __init__(self, is_send, is_send_image, is_inspect_image, is_clusterd):
        self.is_send = is_send
        self.is_send_image = is_send_image
        self.is_inspect_image = is_inspect_image
        self.is_clusterd = is_clusterd
        self.range_resend = None
        self.path_image = None
        self.num_frames = None


def get_image_clustered(is_tobeSent, cf_for_m5):

    depth = 6
    if cf_for_m5.path_image == None:
        path_image = cf.pathVideo_
    else:
        path_image = cf_for_m5.path_image

    cf_getBC = CF_getBC(leftup=[0, 0], rightdown=[cf.imageHeight, cf.imageWidth], depth=depth,
                        path_video=path_image, i_frame=None, shape_or_pose="pose")

    if cf.arg == "0":
        host_ip = '192.168.168.129'
        range_view = range(0, 4)
        range_view_received = range(4, 8)
    elif cf.arg == "1":
        host_ip = '192.168.168.138'
        range_view = range(4, 8)
        range_view_received = range(0, 4)
    else:
        assert False, "PC"


    if is_tobeSent:
        s = connect_to_server((host_ip, 10401))

    path_to_sent = path_image
    if cf_for_m5.range_resend == None:
        if cf_for_m5.num_frames == None:
            range_frame = range(150, 600)
        else:
            range_frame = range(0, cf_for_m5.num_frames)
    else:
        range_frame = []
        for i in range(cf_for_m5.range_resend.shape[0]):
            range_frame.append(cf_for_m5.range_resend[i])

    if cf_for_m5.is_inspect == True:
        print("Inspecting")
        function.f4_inspect_received_files.inspect_image(range_frame=range_frame, range_view=range_view_received, cf_for_m5=cf_for_m5)

    if cf_for_m5.is_clusterd:
        for i_frame in range_frame:
            for i_view in range_view:
                begin = datetime.datetime.now()
                # cluster
                cf_getBC.i_frame = i_frame

                pathFile_list = [path_image + "v%d/image%d_%s.txt" % (i_view, cf_getBC.i_frame, "B"),
                                 path_image + "v%d/image%d_%s.txt" % (i_view, cf_getBC.i_frame, "C")]
                if 0:
                    if (not os.path.exists(pathFile_list[0])) or (not os.path.exists(pathFile_list[1])):
                        get_B_C(i_view, cf_getBC)
                else:
                    get_B_C(i_view, cf_getBC)

                while is_tobeSent:
                    for i_file in ["B"]:
                        try:
                            send_a_file(s, path_image + "v%d/image%d_%s.txt" % (i_view, cf_getBC.i_frame, i_file))
                            break
                        except:
                            print(utility.str2red("Error in sending. v%d, image%d, file_%s" % (i_view, i_frame, i_file)))
                            s = connect_to_server(('192.168.168.138', 10401))

                while is_tobeSent:
                    try:
                        send_a_file(s, path_image + "v%d/image%d_C.txt" % (i_view, cf_getBC.i_frame))
                        break
                    except:
                        print(utility.str2red("Error in sending. v%d, image%d, file_%s" % (i_view, i_frame, "C")))
                        s = connect_to_server(('192.168.168.138', 10401))

                print(("image: %d, view: %d, Time of producing 1 cluster of image: " % (i_frame, i_view)) + str((datetime.datetime.now() - begin)))
            print("finish sending i_frame: %d" % i_frame)

    if cf_for_m5.is_send_image == True:
        for i_frame in range_frame:
            for i_view in range_view:
                # image
                if 1:
                    if 1:
                        while True:
                            try:
                                pathFile_image = path_to_sent + "%d_bmp/%d-%d.bmp" % (i_view, i_frame, i_view)
                                if not os.path.exists(pathFile_image):
                                    print("Not exists: %s" % pathFile_image)
                                send_a_file2(s, pathFile_image, path_to_sent)
                                # send_a_file(s, pathFile_image)
                                break
                            except NameError:
                                raise NameError
                            except (ConnectionAbortedError, ConnectionResetError):
                                print(utility.str2red("Error in sending: %s, retrying..." % (pathFile_image)))
                                # print(utility.str2red("Error in sending. v%d, image%d" % (i_view, i_frame)))
                                s = connect_to_server((host_ip, 10401))
                    else:
                        pathFile_image = path_to_sent + "%d/%d-%d.bmp" % (i_view, i_frame, i_view)
                        if not os.path.exists(pathFile_image):
                            print("Not exists: %s" % pathFile_image)
                        send_a_file2(s, pathFile_image, path_to_sent)
            print("finish sending i_frame: %d" % i_frame)


def get_image_clustered2(is_tobeSent, cf_for_m5):

    depth = 6
    if cf_for_m5.path_image == None:
        path_image = cf.pathVideo_
    else:
        path_image = cf_for_m5.path_image

    cf_getBC = CF_getBC(leftup=[0, 0], rightdown=[cf.imageHeight, cf.imageWidth], depth=depth,
                        path_video=path_image, i_frame=None, shape_or_pose="pose")

    if cf.arg == "0":
        host_ip = '192.168.168.129'
        range_view = range(0, 4)
        range_view_received = range(4, 8)
    elif cf.arg == "1":
        host_ip = '192.168.168.138'
        range_view = range(4, 8)
        range_view_received = range(0, 4)
    else:
        assert False, "PC"


    if is_tobeSent:
        s = connect_to_server((host_ip, 10401))

    path_to_sent = path_image
    if cf_for_m5.range_resend == None:
        if cf_for_m5.num_frames == None:
            range_frame = range(150, 600)
        else:
            range_frame = range(0, cf_for_m5.num_frames)
    else:
        range_frame = []
        for i in range(cf_for_m5.range_resend.shape[0]):
            range_frame.append(cf_for_m5.range_resend[i])

    if cf_for_m5.is_inspect_image == True:
        print("Inspecting")
        function.f4_inspect_received_files.inspect_image(range_frame=range_frame, range_view=range_view_received, cf_for_m5=cf_for_m5)

    if cf_for_m5.is_clusterd:
        for i_frame in range_frame:
            for i_view in range_view:
                begin = datetime.datetime.now()
                # cluster .
                cf_getBC.i_frame = i_frame

                pathFile_list = [path_image + "v%d/image%d_%s.txt" % (i_view, cf_getBC.i_frame, "B"),
                                 path_image + "v%d/image%d_%s.txt" % (i_view, cf_getBC.i_frame, "C")]
                if 1:
                    if (not os.path.exists(pathFile_list[0])) or (not os.path.exists(pathFile_list[1])):
                        get_B_C(i_view, cf_getBC)
                else:
                    get_B_C(i_view, cf_getBC)

                # num_cluster = {"B": -1, "C": -1}
                for i_file in ["B", "C"]:
                    while is_tobeSent:
                        try:
                            send_a_file(s, path_image + "v%d/image%d_%s.txt" % (i_view, cf_getBC.i_frame, i_file))
                            break
                        except:
                            print(utility.str2red("Error in sending. v%d, image%d, file_%s" % (i_view, i_frame, i_file)))
                            s = connect_to_server(('192.168.168.138', 10401))

                print(("image: %d, view: %d, Time of producing 1 cluster of image: " % (i_frame, i_view)) + str((datetime.datetime.now() - begin)))
            print("finish i_frame: %d" % i_frame)

    if cf_for_m5.is_send_image == True:
        for i_frame in range_frame:
            for i_view in range_view:
                # image
                if 1:
                    if 1:
                        while True:
                            try:
                                pathFile_image = path_to_sent + "%d/%d-%d.bmp" % (i_view, i_frame, i_view)
                                if not os.path.exists(pathFile_image):
                                    print("Not exists: %s" % pathFile_image)
                                send_a_file2(s, pathFile_image, path_to_sent)
                                # send_a_file(s, pathFile_image)
                                break
                            except NameError:
                                raise NameError
                            except (ConnectionAbortedError, ConnectionResetError):
                                print(utility.str2red("Error in sending: %s, retrying..." % (pathFile_image)))
                                # print(utility.str2red("Error in sending. v%d, image%d" % (i_view, i_frame)))
                                s = connect_to_server((host_ip, 10401))
                    else:
                        pathFile_image = path_to_sent + "%d/%d-%d.bmp" % (i_view, i_frame, i_view)
                        if not os.path.exists(pathFile_image):
                            print("Not exists: %s" % pathFile_image)
                        send_a_file2(s, pathFile_image, path_to_sent)
            print("finish sending i_frame: %d" % i_frame)


if __name__ == "__main__":
    # inspect
    if 0:
        cf_for_m5 = CF_for_m5(is_inspect=True, is_clusterd=False)
        get_image_clustered(is_tobeSent=False, cf_for_m5=cf_for_m5)

    # resend
    if 1:
        cf_for_m5 = CF_for_m5(is_inspect=False, is_clusterd=False)
        print(os.path.dirname(__file__))
        cf_for_m5.range_resend = np.loadtxt("./image_request_resent.txt", dtype=np.int32)
        get_image_clustered(is_tobeSent=True, cf_for_m5=cf_for_m5)
