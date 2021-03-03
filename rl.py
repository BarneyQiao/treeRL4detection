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



