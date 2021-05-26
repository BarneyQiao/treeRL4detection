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
import time




ON_TRAIN = True
# ON_TRAIN = False

# set env
env = MediaEnv()
s_dim = env.state_dim
a_dim = env.action_dim

# load data
files, labels = ut.get_img_boxes()
ut.MAX_EPISODES = len(files)

# set RL method
rl = DQN(a_dim, s_dim)

steps = []
def train():
    rl.restore() # retrain
    ut.EPS = 0.1
    for epoch in range(ut.MAX_EPOCHES):
        epoch_r = 0
        print('Epoch：{0}'.format(epoch))
        eps = max(ut.EPS- epoch *0.1,0.18)
        # epoch级别
        print(eps)
        # start training
        for i in range(ut.MAX_EPISODES):
            s = env.reset(index=i,files=files,labels=labels)
            # env.render()
            ep_r = 0.
            for j in range(ut.MAX_EP_STEPS):
                if epoch >=0:
                    env.render()

                a = rl.choose_action(s,eps)

                s_, r, done = env.step(a)
                rl.store_transition(s, a, r, s_)
                if done and j < ut.MAX_EP_STEPS-1:
                    a = 13 #如果done
                    rl.store_transition(s_, a, r, s_) # 多存一个记忆

                ep_r += r
                if rl.memory_full and j %1==0:
                    # start to learn once has fulfilled the memory
                    rl.learn()
                    # print('learn')

                # 树结构
                if j%test_steps ==0:
                    s = env.reset(index=i,files=files,labels=labels)
                else:
                    s = s_
                if done or j == ut.MAX_EP_STEPS-1:
                    epoch_r += ep_r
                    print('Ep:%i - %i | %s | ep_r: %.1f | step: %i | epoch_r: %i' % (epoch,i, '---' if not done else 'done', ep_r, j,epoch_r))
                    break
        rl.save(str(epoch))



test_steps = 10
def eval():
    rl.restore()
    # env.viewer.set_vsync(True)
    files, labels = ut.get_img_boxes()
    ut.MAX_EPISODES = len(files)
    # rl.Q_local.eval()
    for i in range(ut.MAX_EPISODES):
        s = env.reset(index=i, files=files, labels=labels)
        env.render()
        ep_r = 0.
        for j in range(500):
            env.render()

            a = rl.choose_action_test(s)

            s_, r, done = env.step(a)

            # rl.store_transition(s, a, r, s_)

            ep_r += r
            # 树结构
            if j % test_steps == 0:
                s = env.reset(index=i, files=files, labels=labels)
            else:
                s = s_

            if done or j == 500 - 1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                time.sleep(3)
                break


if ON_TRAIN:
    train()
else:
    eval()



