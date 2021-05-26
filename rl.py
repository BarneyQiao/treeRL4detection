# -*- coding: utf-8 -*-
"""
@Time          : 2021/3/3 9:18
@Author        : BarneyQ
@File          : rl.py
@Software      : PyCharm
@Description   :
@Modification  :
    @Author    :
    @Time      :
    @Detail    :

"""
import utilds_rf as ut
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import os
import random

GPU = 0
device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

class DQN(object):
    def __init__(self, a_dim, s_dim,max_episodes):

        self.a_dim = a_dim
        self.s_dim = s_dim
        self.lr = ut.LEARNING_RATE
        self.batch_size = ut.BATCH_SIZE
        self.gamma = ut.GAMMA
        self.Q_local =  Net(a_dim=self.a_dim,s_dim=self.s_dim).to(device)
        self.Q_target =  Net(a_dim=self.a_dim,s_dim=self.s_dim).to(device)

        self.memory_full = False # 默认记忆库未满

        # self.memory = deque(maxlen=max_episodes * ut.MAX_EP_STEPS) #记忆库数量
        self.memory = deque(maxlen=10000) #记忆库数量

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.Q_local.parameters(), lr=self.lr)


    def choose_action(self, state,eps):
        rand = np.random.rand()
        # eps = 0.1
        if rand < eps:  # random policy
            action = np.random.randint(0, self.a_dim) # 随机一个动作
            # sec_eps = np.random.rand()
            # if sec_eps <0.5:
            #     action_scale = np.random.randint(0, 5)
            #     action = action_scale
            # else:
            #     action_trans = np.random.randint(5, self.a_dim-1)
            #     # action_trans += 5
            #     action = action_trans
        else:  # greedy policy
            state = torch.FloatTensor(state).to(device)
            # state = torch.from_numpy(state)
            action_value = self.Q_local.forward(state)
            # 拆分两组,每组最高的 随机跳出来
            action_value = torch.squeeze(action_value)
            action_scale_value = action_value[0:5]
            action_trans_value = action_value[5:self.a_dim]

            # action_scale_value_max = sum(action_scale_value).cpu().item()
            # action_trans_value_max = sum(action_trans_value).cpu().item()

            # if action_scale_value_max > action_trans_value_max:
            #     action_scale = torch.max(action_scale_value, 0)[1].cpu().item()
            #     action = action_scale
            # else:
            #     action_trans = torch.max(action_trans_value, 0)[1].cpu().item()
            #     action_trans +=5
            #     action = action_trans


            if np.random.rand() < 0.5:
                action_scale = torch.max(action_scale_value, 0)[1].cpu().item()
                action = action_scale
            else:
                action_trans = torch.max(action_trans_value, 0)[1].cpu().item()
                action_trans +=5
                action = action_trans
        return action


    def choose_action_rf(self, state,eps):
        rand = np.random.rand()
        # eps = 0.1
        if rand < eps:  # random policy
            # 加一个 k >0.5 选择带来正reward的动作




            action = np.random.randint(0, self.a_dim-1) # 随机一个动作，不包括trigger

        else:  # greedy policy
            state = torch.FloatTensor(state).to(device)
            # state = torch.from_numpy(state)
            action_value = self.Q_local.forward(state)
            # 拆分两组,每组最高的 随机跳出来
            action_value = torch.squeeze(action_value)
            action = torch.max(action_value, 0)[1].cpu().item()
        return action

# 测试用的动作选择，直接都选值最大的
    def choose_action_test(self, state):

        self.Q_local.eval()

        state = torch.FloatTensor(state).to(device)
        # state = torch.from_numpy(state)
        action_value = self.Q_local.forward(state)
        action_value = torch.squeeze(action_value)

        action = torch.max(action_value, 0)[1].cpu().item()


        return action

    def learn(self,f):
        experiences = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        Q_values = self.Q_local(states)
        Q_values = torch.gather(input=Q_values, dim=1, index=actions)
        Q_targets = self.Q_target(next_states).detach()
        # Q_targets, _ = torch.max(input=Q_targets, dim=-1, keepdim=True)
        if f ==True :
            Q_targets = rewards
        else:
            Q_targets = rewards + self.gamma  * Q_targets.max(1)[0].view(self.batch_size,1)

        loss = self.loss_func(Q_values, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss




    def store_transition(self, s, a, r, s_):
        # transition = np.hstack((s, a, r, s_))
        self.memory.append((s, a, r, s_))
        if len(self.memory) >= self.memory.maxlen:
            self.memory_full = True


    def save(self,epoch):
        path_state_dict = './' + 'para'+'-epoch'+epoch + '.pth'
        state_dic = self.Q_local.state_dict()
        torch.save(state_dic, path_state_dict)
        print('save model param!')

    def restore(self):
        path = './para-epoch300.pth'
        if os.path.exists(path):
            self.Q_local.load_state_dict(torch.load(path))
            self.Q_target.load_state_dict(torch.load(path))
            print("load model param!")

#DQN 网络
class Net(nn.Module):
    def __init__(self, a_dim, s_dim):
        super(Net, self).__init__()
    #     self.dqn = nn.Sequential(
    #         nn.Linear(s_dim, 4096),
    #         nn.ReLU(True),
    #         nn.Linear(4096,1024),
    #         nn.ReLU(True),
    #         nn.Linear(1024,1024),
    #         nn.ReLU(True),
    #         nn.Linear(1024,1024),
    #         nn.ReLU(True),
    # )
        self.dqn = nn.Sequential(
                    nn.Linear(s_dim, 512),
                    nn.ReLU(True),
                    nn.Linear(512,512),
                    nn.ReLU(True),
                    # nn.Linear(512, 512),
                    # nn.ReLU(True),

            )
        self.out = nn.Linear(512, a_dim)
        self.initialize() # 初始化
        # print('ss')

    def forward(self, x):
        x = self.dqn(x)
        actions_value = self.out(x)
        return actions_value

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

if __name__ == '__main__':
    net = Net(13,59000)
    print(net)