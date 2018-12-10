#-*- coding: UTF-8 -*-
import socket,time,socketserver,struct,os
import utility
# host='JIANGPEIYUANVISG'
host = socket.gethostname()
port=10401
ADDR=(host,port)

class MyRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):

        print('connected from:', self.client_address)

        while True:

            # 定义文件信息。128s表示文件名为128bytes长，l表示一个int或log文件类型，在此为文件大小
            fileinfo_size=struct.calcsize('128sl')
            self.buf = self.request.recv(fileinfo_size)

            # 如果不加这个if，第一个文件传输完成后会自动走到下一句
            if self.buf:

                # 根据128sl解包文件信息，与client端的打包规则相同
                self.str_received, self.filesize =struct.unpack('128sl',self.buf)

                # # 文件名长度为128，大于文件名实际长度
                # print('filesize is: ',self.filesize,'filename size is: ',len(self.filename))
                # # 使用strip()删除打包时附加的多余空字符

                [self.filename, self.pathSave] = self.str_received.decode().split("##")
                pathSave = (self.pathSave).strip('\00')
                utility.mkdir(pathSave)
                for i_view in range(8):
                    pathSave_view = pathSave + "%d_bmp/" % i_view
                    utility.mkdir(pathSave_view)
                    pathSave_view = pathSave + "v%d/" % i_view
                    utility.mkdir(pathSave_view)

                self.filenewname = os.path.join(pathSave, (self.filename).strip('\00'))
                print(self.filenewname,type(self.filenewname))

                # 定义接收了的文件大小
                recvd_size = 0
                file = open(self.filenewname,'wb')
                # print('stat receiving...')
                while not recvd_size == self.filesize:
                    if self.filesize - recvd_size > 1024:
                        rdata = self.request.recv(1024)
                        recvd_size += len(rdata)
                    else:
                        rdata = self.request.recv(self.filesize - recvd_size)
                        recvd_size = self.filesize
                    file.write(rdata)
                file.close()
                # print('receive done')
        #self.request.close()

tcpServ = socketserver.ThreadingTCPServer(ADDR, MyRequestHandler)
print(tcpServ.socket)
print('waiting for connection...' )
tcpServ.serve_forever()