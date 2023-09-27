import socket
import struct
import threading
import time
import cv2
import numpy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


ENCODE_PARAMETER_DICT = {(0.125, 1, 43): 0, (0.125, 0.5, 43): 1, (0.125, 1, 38): 2, (0.125, 0.25, 43): 3,
                         (0.25, 1, 43): 4, (0.125, 0.5, 38): 5, (0.125, 0.125, 43): 6, (0.125, 1, 33): 7,
                         (0.125, 0.25, 38): 8, (0.25, 0.5, 43): 9, (0.125, 0.0625, 43): 10, (0.25, 1, 38): 11,
                         (0.125, 0.125, 38): 12, (0.125, 0.5, 33): 13, (0.125, 1, 28): 14, (0.25, 0.25, 43): 15,
                         (0.125, 0.0625, 38): 16, (0.125, 0.25, 33): 17, (0.25, 0.125, 43): 18, (0.5, 1, 43): 19,
                         (0.25, 0.5, 38): 20, (0.125, 1, 23): 21, (0.125, 0.125, 33): 22, (0.125, 0.5, 28): 23,
                         (0.25, 1, 33): 24, (0.25, 0.0625, 43): 25, (0.125, 0.25, 28): 26, (0.125, 0.0625, 33): 27,
                         (0.25, 0.25, 38): 28, (0.5, 0.5, 43): 29, (0.125, 0.5, 23): 30, (0.125, 0.125, 28): 31,
                         (0.5, 1, 38): 32, (0.25, 0.125, 38): 33, (0.125, 0.25, 23): 34, (0.125, 0.0625, 28): 35,
                         (0.75, 1, 43): 36, (0.25, 1, 28): 37, (0.5, 0.25, 43): 38, (0.25, 0.5, 33): 39,
                         (0.125, 0.125, 23): 40, (0.25, 0.0625, 38): 41, (0.125, 0.0625, 23): 42, (0.25, 0.25, 33): 43,
                         (0.5, 0.125, 43): 44, (0.75, 0.5, 43): 45, (0.5, 0.5, 38): 46, (0.25, 1, 23): 47,
                         (1, 1, 43): 48, (0.5, 1, 33): 49, (0.25, 0.125, 33): 50, (0.75, 1, 38): 51,
                         (0.25, 0.5, 28): 52, (0.5, 0.0625, 43): 53, (0.75, 0.25, 43): 54, (0.5, 0.25, 38): 55,
                         (0.25, 0.0625, 33): 56, (0.25, 0.25, 28): 57, (1, 0.5, 43): 58, (0.25, 0.125, 28): 59,
                         (0.25, 0.5, 23): 60, (0.75, 0.125, 43): 61, (0.5, 0.125, 38): 62, (1, 1, 38): 63,
                         (0.75, 0.5, 38): 64, (0.25, 0.0625, 28): 65, (0.5, 1, 28): 66, (0.25, 0.25, 23): 67,
                         (1, 0.25, 43): 68, (0.75, 1, 33): 69, (0.5, 0.5, 33): 70, (0.5, 0.0625, 38): 71,
                         (0.75, 0.0625, 43): 72, (0.25, 0.125, 23): 73, (0.75, 0.25, 38): 74, (0.25, 0.0625, 23): 75,
                         (0.5, 0.25, 33): 76, (1, 0.125, 43): 77, (1, 0.5, 38): 78, (0.5, 1, 23): 79,
                         (0.75, 0.125, 38): 80, (1, 1, 33): 81, (0.5, 0.125, 33): 82, (1, 0.0625, 43): 83,
                         (0.75, 1, 28): 84, (0.5, 0.5, 28): 85, (1, 0.25, 38): 86, (0.5, 0.0625, 33): 87,
                         (0.75, 0.5, 33): 88, (0.75, 0.0625, 38): 89, (0.5, 0.25, 28): 90, (0.75, 0.25, 33): 91,
                         (1, 0.125, 38): 92, (0.5, 0.125, 28): 93, (1, 1, 28): 94, (0.5, 0.5, 23): 95,
                         (0.5, 0.0625, 28): 96, (0.75, 1, 23): 97, (0.75, 0.125, 33): 98, (1, 0.5, 33): 99,
                         (1, 0.0625, 38): 100, (0.5, 0.25, 23): 101, (0.75, 0.5, 28): 102, (0.75, 0.0625, 33): 103,
                         (0.5, 0.125, 23): 104, (1, 0.25, 33): 105, (0.5, 0.0625, 23): 106, (0.75, 0.25, 28): 107,
                         (1, 1, 23): 108, (1, 0.125, 33): 109, (0.75, 0.125, 28): 110, (0.75, 0.0625, 28): 111,
                         (1, 0.0625, 33): 112, (0.75, 0.5, 23): 113, (1, 0.5, 28): 114, (0.75, 0.25, 23): 115,
                         (1, 0.25, 28): 116, (0.75, 0.125, 23): 117, (1, 0.125, 28): 118, (0.75, 0.0625, 23): 119,
                         (1, 0.0625, 28): 120, (1, 0.5, 23): 121, (1, 0.25, 23): 122, (1, 0.125, 23): 123,
                         (1, 0.0625, 23): 124}

DECODE_PARAMETER_DICT = {0: (0.125, 1, 43), 1: (0.125, 0.5, 43), 2: (0.125, 1, 38), 3: (0.125, 0.25, 43), 
       4: (0.25, 1, 43), 5: (0.125, 0.5, 38), 6: (0.125, 0.125, 43), 7: (0.125, 1, 33), 
       8: (0.125, 0.25, 38), 9: (0.25, 0.5, 43), 10: (0.125, 0.0625, 43), 11: (0.25, 1, 38), 
       12: (0.125, 0.125, 38), 13: (0.125, 0.5, 33), 14: (0.125, 1, 28), 15: (0.25, 0.25, 43), 
       16: (0.125, 0.0625, 38), 17: (0.125, 0.25, 33), 18: (0.25, 0.125, 43), 19: (0.5, 1, 43), 
       20: (0.25, 0.5, 38), 21: (0.125, 1, 23), 22: (0.125, 0.125, 33), 23: (0.125, 0.5, 28), 
       24: (0.25, 1, 33), 25: (0.25, 0.0625, 43), 26: (0.125, 0.25, 28), 27: (0.125, 0.0625, 33), 
       28: (0.25, 0.25, 38), 29: (0.5, 0.5, 43), 30: (0.125, 0.5, 23), 31: (0.125, 0.125, 28), 
       32: (0.5, 1, 38), 33: (0.25, 0.125, 38), 34: (0.125, 0.25, 23), 35: (0.125, 0.0625, 28), 
       36: (0.75, 1, 43), 37: (0.25, 1, 28), 38: (0.5, 0.25, 43), 39: (0.25, 0.5, 33), 
       40: (0.125, 0.125, 23), 41: (0.25, 0.0625, 38), 42: (0.125, 0.0625, 23), 43: (0.25, 0.25, 33), 
       44: (0.5, 0.125, 43), 45: (0.75, 0.5, 43), 46: (0.5, 0.5, 38), 47: (0.25, 1, 23), 
       48: (1, 1, 43), 49: (0.5, 1, 33), 50: (0.25, 0.125, 33), 51: (0.75, 1, 38), 
       52: (0.25, 0.5, 28), 53: (0.5, 0.0625, 43), 54: (0.75, 0.25, 43), 55: (0.5, 0.25, 38), 
       56: (0.25, 0.0625, 33), 57: (0.25, 0.25, 28), 58: (1, 0.5, 43), 59: (0.25, 0.125, 28), 
       60: (0.25, 0.5, 23), 61: (0.75, 0.125, 43), 62: (0.5, 0.125, 38), 63: (1, 1, 38), 
       64: (0.75, 0.5, 38), 65: (0.25, 0.0625, 28), 66: (0.5, 1, 28), 67: (0.25, 0.25, 23), 
       68: (1, 0.25, 43), 69: (0.75, 1, 33), 70: (0.5, 0.5, 33), 71: (0.5, 0.0625, 38), 
       72: (0.75, 0.0625, 43), 73: (0.25, 0.125, 23), 74: (0.75, 0.25, 38), 75: (0.25, 0.0625, 23), 
       76: (0.5, 0.25, 33), 77: (1, 0.125, 43), 78: (1, 0.5, 38), 79: (0.5, 1, 23), 
       80: (0.75, 0.125, 38), 81: (1, 1, 33), 82: (0.5, 0.125, 33), 83: (1, 0.0625, 43), 
       84: (0.75, 1, 28), 85: (0.5, 0.5, 28), 86: (1, 0.25, 38), 87: (0.5, 0.0625, 33), 
       88: (0.75, 0.5, 33), 89: (0.75, 0.0625, 38), 90: (0.5, 0.25, 28), 91: (0.75, 0.25, 33), 
       92: (1, 0.125, 38), 93: (0.5, 0.125, 28), 94: (1, 1, 28), 95: (0.5, 0.5, 23), 
       96: (0.5, 0.0625, 28), 97: (0.75, 1, 23), 98: (0.75, 0.125, 33), 99: (1, 0.5, 33), 
       100: (1, 0.0625, 38), 101: (0.5, 0.25, 23), 102: (0.75, 0.5, 28), 103: (0.75, 0.0625, 33), 
       104: (0.5, 0.125, 23), 105: (1, 0.25, 33), 106: (0.5, 0.0625, 23), 107: (0.75, 0.25, 28), 
       108: (1, 1, 23), 109: (1, 0.125, 33), 110: (0.75, 0.125, 28), 111: (0.75, 0.0625, 28), 
       112: (1, 0.0625, 33), 113: (0.75, 0.5, 23), 114: (1, 0.5, 28), 115: (0.75, 0.25, 23), 
       116: (1, 0.25, 28), 117: (0.75, 0.125, 23), 118: (1, 0.125, 28), 119: (0.75, 0.0625, 23), 
       120: (1, 0.0625, 28), 121: (1, 0.5, 23), 122: (1, 0.25, 23), 123: (1, 0.125, 23), 
       124: (1, 0.0625, 23)}


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim=128):
        super(Actor, self).__init__()

        self.s_dim = state_dim  # [s_info, s_len]
        self.a_dim = action_dim
        self.num_hids = mid_dim

        # layer define
        self.fc1 = nn.Linear(1, self.num_hids)
        self.conv1 = nn.Conv1d(1, self.num_hids, kernel_size=4)
        self.conv2 = nn.Conv1d(3, self.num_hids, kernel_size=1)

        self.out_linear = nn.Linear(2816, self.a_dim)

    def merge_net(self, inputs):
        inputs = torch.reshape(inputs, (-1, self.s_dim[0], self.s_dim[1]))
        # 1-average network throughput
        split_0 = F.relu(self.conv1(inputs[:, 0:1, :].view(-1, 1, self.s_dim[1])))
        # 2-camera buffer size
        split_1 = F.relu(self.fc1(inputs[:, 1:2, -1]))
        # 3 - past segment upload time
        split_2 = F.relu(self.conv1(inputs[:, 2:3, :].view(-1, 1, self.s_dim[1])))
        # 4-past segment size
        split_3 = F.relu(self.conv1(inputs[:, 3:4, :].view(-1, 1, self.s_dim[1])))
        # 5,6,7-past segment FR, QP, RS
        split_4 = F.relu(self.conv2(inputs[:, 4:7, -1].view(-1, 3, 1)))
        # 8- past content dynamics
        split_5 = F.relu(self.conv1(inputs[:, 7:8, :].view(-1, 1, self.s_dim[1])))

        # flatten
        split_0_flatten, split_2_flatten, split_3_flatten, split_4_flatten, split_5_flatten = split_0.flatten(
            start_dim=1), split_2.flatten(start_dim=1), split_3.flatten(start_dim=1), split_4.flatten(
            start_dim=1), split_5.flatten(start_dim=1)

        # merge
        merge_net = torch.cat(
            [split_0_flatten, split_1, split_2_flatten, split_3_flatten, split_4_flatten, split_5_flatten], dim=1)

        return merge_net

    def forward(self, inputs):
        merge_net = self.merge_net(inputs)

        # feature map rl
        self.featuremap_rl = merge_net.detach()

        action_head = self.out_linear(merge_net)

        action_prob = F.relu(action_head)

        return action_prob


s_dim = [8, 8]
a_dim = 125
state = torch.ones(1, 8, 8)
state[0, 0] = 5
state[0, 1] = 0
state[0, 2] = 1
state[0, 3] = 10
state[0, 4] = 1
state[0, 5] = 1
state[0, 6] = 23
state[0, 7] = 0
pre_action = 108
pre_file_size = 0

def config(bandwidth, buffer_size, trans_time, video_chunk_size, action, content_change_rate = 0.0):
    # 1- input
    state[:, :, :-1] = state[:, :, 1:]
    state[0, 0, -1] = bandwidth
    state[0, 1, -1] = buffer_size
    state[0, 2, -1] = trans_time
    state[0, 3, -1] = video_chunk_size

    fps, res, qp = DECODE_PARAMETER_DICT[action]
    state[0, 4, -1] = res
    state[0, 5, -1] = fps
    state[0, 6, -1] = qp
    state[0, 7, -1] = content_change_rate
    print("state: ", state)
    
    # 2- load model
    model_path = './casva_model.pth'
    agent = Actor(state_dim=s_dim, action_dim=a_dim)
    agent.load_state_dict(torch.load(model_path))
    
    # 3-prediction, output action
    logits = agent(torch.FloatTensor(state))
    prediction = torch.argmax(logits, dim=-1)
    print("prediction: ", prediction)
    return prediction.item()


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
    trans_time = 0.0  # s

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
        trans_time += end - start
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

    # 4- send action
    file_size = os.path.getsize(videoPath)  # Byte
    bandwidth = file_size / trans_time / 1024 / 1024   # MB/s
    video_chunk_size = file_size * 8 / 1024 / 1024  # Mb
    new_action = pre_action
    content_change_rate = 0.0
    if pre_file_size == 0:
        content_change_rate = 0.0
        pre_file_size = file_size
    else:
        content_change_rate = (file_size - pre_file_size) / pre_file_size
        pre_file_size = file_size
    ret = config(bandwidth, 0, trans_time, video_chunk_size, new_action, content_change_rate)

    # 5- send action


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
