import socket
import struct
import threading
import time
import cv2
import numpy


def recvByCount(sock, count):  # 读取count长度的数据
    buf = b''
    while count:
        newbuf = sock.recv(count)  # s.recv()接收. 因为client是通过 sk.recv()来进行接受数据，而count表示，最多每次接受count字节
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def receiveVideo(sock, addr):
    # 接受图片及大小的信息
    print('connect from:' + str(addr))
    videoName = "%s.h264" % (time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())))
    videoPath = "./tcp_h264/" + videoName
    print("videoName: " , videoName)
    f = open(videoPath, 'wb')  # 以二进制追加模式打开文件

    while not getattr(sock, '_closed'):
        start = time.time()
        # 接收图片帧大小
        recvSize = struct.calcsize('i')
        size_bytes = recvByCount(sock, recvSize)
        if size_bytes is None:
            break
        size = int.from_bytes(size_bytes, byteorder='big')
        print('the image size = ', size, 'bytes,   size_bytes = ', " ".join([hex(int(i)) for i in size_bytes]),
              ',     len(size_bytes) = ', len(size_bytes))
        if len(size_bytes) == 0 or size == 0:
            continue
        # 接收图片
        imageData = recvByCount(sock, size)
        if imageData is None:
            break
        print('imageData.length = ', len(imageData))

        end = time.time()
        print('receiving the image costs time = ', end - start)

        f.write(imageData)  # 写入文件
        f.flush()  # 刷新缓冲区

    f.close()


if __name__ == '__main__':
    address = ('0.0.0.0', 6010)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(address)
    s.listen(5)
    print(f"[*]Listening:  {address}")
    while True:
        sock, addr = s.accept()
        # 创建新线程来处理TCP连接:
        t = threading.Thread(target=receiveVideo, args=(sock, addr))
        t.start()
        t.join()
