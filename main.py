# -*- coding: utf-8 -*-
"""
@Time          : 2021/3/3 9:18
@Author        : BarneyQ
@File          : main.py
@Software      : PyCharm
@Description   :
@Modification  :
    @Author    :
    @Time      :
    @Detail    :

"""
"""
Build the basic framework for main.py, rl.py and env.py.
"""
from env import MediaEnv
from rl import DQN
import utils as ut



# set env
env = MediaEnv()
s_dim = env.state_dim
a_dim = env.action_dim

# set RL method
rl = DQN(a_dim, s_dim)

# start training
for i in range(ut.MAX_EPISODES):
    s = env.reset()
    for j in range(ut.MAX_EP_STEPS):
        env.render()

        a = rl.choose_action(s)

        s_, r, done = env.step(a)

        rl.store_transition(s, a, r, s_)

        if rl.memory_full:
            # start to learn once has fulfilled the memory
            rl.learn()

        s = s_





