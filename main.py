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




ON_TRAIN = True

# set env
env = MediaEnv()
s_dim = env.state_dim
a_dim = env.action_dim

# set RL method
rl = DQN(a_dim, s_dim)

steps = []
def train():
    # start training
    for i in range(ut.MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(ut.MAX_EP_STEPS):
            # env.render()

            a = rl.choose_action(s)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == ut.MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done = env.step(a)


if ON_TRAIN:
    train()
else:
    eval()



