# -*- coding: utf-8 -*-
"""
@Time          : 2021/3/10 14:11
@Author        : BarneyQ
@File          : main_rf.py
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
from env_rf import MediaEnv
from rl import DQN
import utilds_rf as ut
import time
import numpy as np
import torch

GPU = 0
device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")




# ON_TRAIN = True
ON_TRAIN = False

# set env
env = MediaEnv()
s_dim = env.state_dim
a_dim = env.action_dim

# load data
files, labels = ut.get_img_boxes()
max_episodes = len(files)

# set RL method
rl = DQN(a_dim, s_dim,max_episodes)

steps = []
test_ep = 0

def train_old():
    # rl.restore() # retrain
    # ut.EPS = 0.1
    # store = 0
    for epoch in range(ut.MAX_EPOCHES):
        epoch_r = 0
        print('Epoch：{0}'.format(epoch))
        eps = max(ut.EPS- epoch *0.01,0.1)
        # epoch级别
        print(eps)
        # start training
        for i in range(max_episodes):
            s = env.reset(index=i,files=files,labels=labels)
            env.his_actions_deq = ut.init_his_action_deq(env.his_actions_deq,
                                                          action_dim=env.action_dim)
            # env.render()
            ep_r = 0.
            for j in range(ut.MAX_EP_STEPS):
                if epoch >=0:
                    env.render()

                # a = rl.choose_action_rf(s,eps)
                a = choose_action_rf_reward(s,eps) # 加了概率 试一下

                s_, r, done,cur_iou = env.step(a)
                # rl.store_transition(s, a, r, s_)
                # ep_r += r
                if done and j < ut.MAX_EP_STEPS-1:
                    # 如果提早结束，那么就剩下的步骤一直储存停止动作
                    for z in range(ut.MAX_EP_STEPS-1 - j):
                    # for z in range(2):
                        a = 10 #如果done
                        r = 10
                        rl.store_transition(s_, a, r, s_) # 多存一个记忆
                        # store +=1
                        ep_r += r
                        j += 1
                    # s_ = env.reset(index=i, files=files, labels=labels)
                elif cur_iou <env.init_iou:   ## 一个雷
                    s_ = env.reset(index=i, files=files, labels=labels)
                    r = -3.0
                    rl.store_transition(s, a, r, s_)  # 多存一个记忆
                    # store +=1
                    ep_r += r
                else:
                    rl.store_transition(s, a, r, s_)
                    # store +=1
                    ep_r += r

                if rl.memory_full and j %1 == 0:
                    # start to learn once has fulfilled the memory
                    rl.learn()

                    # print('learn')

                # # 树结构
                # if j%test_steps ==0:
                #     s = env.reset(index=i,files=files,labels=labels)
                # else:
                #     s = s_

                # 如果现在的iou已经小于阈值 ，重置
                # if cur_iou <= 0.1:
                #     s_ = env.reset(index=i,files=files,labels=labels)
                #     r = -5.0
                #     rl.store_transition(s, a, r, s_)  # 多存一个记忆
                #     ep_r += r
                # else:
                #     s = s_

                if done or j == ut.MAX_EP_STEPS-1:
                    epoch_r += ep_r
                    print('Train_Ep:%i - %i | %s | ep_r: %.1f | step: %i | epoch_r: %i' % (epoch,i, '---' if not done else 'done', ep_r, j,epoch_r))


                    # print("store: %i"%(store))
                    # print(len(rl.memory))
                    break
        print('++++++++++++++++ Test +++++++++++++++++++')
        rest_er = eval()
        if rest_er > test_ep:
            print(rest_er)
            rl.save(str(epoch))

def train():
    # 最新的
    for epoch in range(ut.MAX_EPOCHES):
        epoch_r = 0
        print('Epoch：{0}'.format(epoch))
        eps = max(ut.EPS- epoch *0.1,0.1)
        # epoch级别
        print(eps)
        # start training
        for i in range(max_episodes):
            # 初始化图片、动作等
            s = env.reset(index=i,files=files,labels=labels)
            env.his_actions_deq = ut.init_his_action_deq(env.his_actions_deq,
                                                          action_dim=env.action_dim)

            ep_r = 0.
            f = False
            j = 0
            while f ==False :
                if epoch >= 0:
                    env.render()
                a = choose_action_rf_reward(s, eps)  # 选择一个动作 at
                s_, r, done, cur_iou = env.step(a)  # 执行动作获得下一个状态

                rl.store_transition(s, a, r, s_)  # 存储记忆
                s = s_
                ep_r += r
                if cur_iou <=0.0:
                    s = env.reset(index=i, files=files, labels=labels)
                    env.his_actions_deq = ut.init_his_action_deq(env.his_actions_deq,
                                                                 action_dim=env.action_dim)
                    # a = 5 # 停下来，提前结束
                    # s = s_
                    # s_, r, done, cur_iou = env.step(a)  # 执行动作获得下一个状态
                    # rl.store_transition(s, a, r, s_)  # 存储记忆


                if j > ut.MAX_EP_STEPS:
                    f = True
                    # if rl.memory_full:
                    if len(rl.memory)>ut.BATCH_SIZE:
                        rl.learn(f)
                # if a == 5 :
                #     # 如果选择的是停止动作
                #     f = True
                # else:
                #     f = False


                j += 1



            epoch_r += ep_r
            print('Train_Ep:%i - %i | %s | ep_r: %.1f | step: %i | epoch_r: %i' % (epoch,i, '---' if not f else 'done', ep_r, j,epoch_r))



        # print('++++++++++++++++ Test +++++++++++++++++++')
        # rest_er = eval()
        # if rest_er > test_ep:
        #     print(rest_er)
        #     rl.save(str(epoch))

    rl.save('30')


test_steps = 10

def eval():
    rl.restore()
    # env.viewer.set_vsync(True)
    files, labels = ut.get_img_boxes()
    ut.MAX_EPISODES = len(files)
    # rl.Q_local.eval()
    ep_r = 0.
    for i in range(ut.MAX_EPISODES):
        s = env.reset(index=i, files=files, labels=labels)
        env.render()
        ep_r = 0.
        for j in range(test_steps):
            env.render()

            a = rl.choose_action_test(s)
            # a = choose_action_rf_reward(s, 0.5)  # 加了概率 试一下
            s_, r, done,_ = env.step(a)

            # rl.store_transition(s, a, r, s_)

            ep_r += r


            if j ==test_steps-1 or done:
                print('Test_Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                time.sleep(2)
                break
    return ep_r



def choose_action_rf_reward(state, eps):
    rand = np.random.rand()
    if rand < eps:  # random policy
        action = np.random.randint(0,rl.a_dim)
        # print('random-action',action)
    else:
        # 选择IOU 最大的动作
        re_ious = []
        for at in range(env.action_dim):
            img_box = env.state_info['img_box'] # 当前框的 坐标
            # 执行动作得到新的框
            img_box = ut.cmt_imgcor_by_action(list_imgcor=img_box, action=at)
            # 去除异常坐标
            img_box = ut.drop_outliers(img_box)

            re_iou = ut.cal_iou(img_box, env.ground_box)
            re_ious.append(re_iou) # 添加iou


        action = re_ious.index(max(re_ious))
        # print('max-action', action)
        # action = np.random.choice(actions)


    return action




if ON_TRAIN:
    train()
else:
    eval()



