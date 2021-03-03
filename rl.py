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
import torch
import torch.nn as nn

GAMMA = 0.9     # reward discount
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(object):
    def __init__(self, a_dim, s_dim):
        pass

    def choose_action(self, s):
        pass

    def learn(self):
        pass

    def store_transition(self, s, a, r, s_):
        pass

#DQN 网络
class Net(nn.Module):
    def __init__(self, a_dim, s_dim):
        super(Net, self).__init__()
        self.dqn = nn.Sequential(
            nn.Linear(s_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024,1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024,1024),
            nn.ReLU(True),
            nn.Dropout()
    )
        self.out = nn.Linear(1024, a_dim)

    def forward(self, x):
        x = self.dqn(x)
        actions_value = self.out(x)
        return actions_value