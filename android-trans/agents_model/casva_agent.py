#!/usr/bin/python
# -*- coding: utf-8 -*-
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



if __name__ == "__main__":
     # 1- input
    """
    state[0, -1] = float(past_chunk_throughput) * BITS_IN_BYTE / B_IN_MB  # byte/sec --> Mbit / s
    state[1, -1] = buffer_size
    state[2, -1] = float(latency)
    state[3, -1] = np.array(video_chunk_size / 1e6)  # Mb
    state[4, -1] = float(res)
    state[5, -1] = float(fps)
    state[6, -1] = float(qp)
    state[7, -1] = float(content_change_rate)
    """
    s_dim = [8, 8]
    a_dim = 125
    state = torch.rand(1, 8, 8)
    state[0, 0] = 5
    state[0, 1] = 0
    state[0, 2] = 1
    state[0, 3] = 10
    state[0, 4] = 1
    state[0, 5] = 1
    state[0, 6] = 23
    state[0, 7] = 0
    print("state: ", state)

    
    # 2- load model
    model_path = './casva_model.pth'
    agent = Actor(state_dim=s_dim, action_dim=a_dim)
    agent.load_state_dict(torch.load(model_path))
    
    # 3-prediction, output action
    logits = agent(torch.FloatTensor(state))
    prediction = torch.argmax(logits, dim=-1)
    print("prediction: ", prediction)
