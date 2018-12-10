# -*- coding: UTF-8 -*-
import socket, os, struct
import datetime

def connect_to_server(ip_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(ip_port)
    return s


def send_a_file(s, filepath):
# def send_a_file(s, filepath, i_view):
    if os.path.isfile(filepath):
        # 定义打包规则
        fileinfo_size = struct.calcsize('128sl')

        # 定义文件头信息，包含文件名和文件大小
        pathEnd = os.path.splitdrive(filepath)
        pathEnd = (filepath.split("/"))[-2]
        fileName = pathEnd + "/" + os.path.basename(filepath)
        # fileName = "v%d/" % i_view + os.path.basename(filepath)
        fileSize = os.stat(filepath).st_size
        fhead = struct.pack('128sl', fileName.encode(), fileSize)
        s.send(fhead)

        # with open(filepath,'rb') as fo: 这样发送文件有问题，发送完成后还会发一些东西过去
        fo = open(filepath, 'rb')
        while True:
            filedata = fo.read(1024)
            if not filedata:
                break

            s.send(filedata)

        fo.close()

        # s.close()


def send_a_file2(s, filepath, pathSave):
# def send_a_file(s, filepath, i_view):
    if os.path.isfile(filepath):
        # 定义打包规则
        fileinfo_size = struct.calcsize('128sl')

        # 定义文件头信息，包含文件名和文件大小
        pathEnd = os.path.splitdrive(filepath)
        pathEnd = (filepath.split("/"))[-2]
        fileName = pathEnd + "/" + os.path.basename(filepath)
        # fileName = "v%d/" % i_view + os.path.basename(filepath)
        fileSize = os.stat(filepath).st_size
        str_sent = "##".join([fileName, pathSave])
        fhead = struct.pack('128sl', str_sent.encode(), fileSize)
        s.send(fhead)

        # with open(filepath,'rb') as fo: 这样发送文件有问题，发送完成后还会发一些东西过去
        fo = open(filepath, 'rb')
        while True:
            filedata = fo.read(1024)
            if not filedata:
                break

            s.send(filedata)

        fo.close()

        # s.close()


if __name__ == "__main__":

    assert False, "aborted"

    s = connect_to_server(('192.168.168.138', 10401))

    begin = datetime.datetime.now()
    for i_file in range(150, 600):
        # txt
        if 0:
            for i_file_name in ["B", "C"]:

                filepath = r"D:\0_image_1811_1_cali\K1_RT1_D6\v0/image%d_%s.txt" % (i_file, i_file_name)

                send_a_file(s, filepath)
                print("Finish sending: %d" % i_file)
        # image
        if 1:
            for i_view in range(4, 8):
                pathFile_image = r"D:/0_image_1811_1_cali/K1_RT1_D4/%d/%d-%d.bmp" % (i_view, i_file, i_view)
                send_a_file(s, pathFile_image)

        print("Finish sending image: %d" % i_file)

    print(datetime.datetime.now() - begin)