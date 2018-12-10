# -*- coding: UTF-8 -*-
import socket, os, struct
import datetime

def connect_to_server(ip_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(ip_port)
    return s


def send_a_file(s, filepath):
    if os.path.isfile(filepath):
        # 定义打包规则
        fileinfo_size = struct.calcsize('128sl')

        # 定义文件头信息，包含文件名和文件大小
        fileName = os.path.basename(filepath)
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

s = connect_to_server(('192.168.168.138', 10401))

begin = datetime.datetime.now()
for i_file in range(150, 234):
    for i_file_name in ["B", "C"]:

        filepath = r"D:\0_image_1810_9_cali\0_image_1810_9_cali_K2_RT1_D12\v0/image%d_%s.txt" % (i_file, i_file_name)

        send_a_file(s, filepath)
        print("Finish sending: %d" % i_file)

print(datetime.datetime.now() - begin)