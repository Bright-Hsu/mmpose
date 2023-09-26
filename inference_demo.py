import socket
import struct
import threading
import time
import cv2
import numpy
import os


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
    videoPath = "./android-trans/tcp_h264/" + videoName
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
    ffmpeg_cmd  = "ffmpeg -i " + videoPath + " -c:v libx264 -c:a copy " + videoPath[:-5] + ".mp4"
    print(ffmpeg_cmd)
    os.system(ffmpeg_cmd)
    new_videoPath = videoPath[:-5] + ".mp4"
    infer_start = time.time()
    infer_cmd = "python demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth --input " + new_videoPath + " --output-root test_results1/"
    print(infer_cmd)
    os.system(infer_cmd)
    infer_end = time.time()
    infer_time = infer_end - infer_start
    print('inferencing the video costs time = ', infer_time)

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
