# -*- coding: utf-8 -*-
"""
@Time          : 2021/3/2 15:44
@Author        : BarneyQ
@File          : main.py
@Software      : PyCharm
@Description   :
@Modification  :
    @Author    :
    @Time      :
    @Detail    :

"""
# 导入环境和学习方法
from environment import Env_Med
from rl import DQN

# 设置全局变量
MAX_EPOCHES = 25
MAX_EPISODES = 500
MAX_EP_STEPS = 200

env = Env_Med()
s_dim = env.state_dim
a_dim = env.action_dim


# 设置学习方法
rl = DQN(a_dim,s_dim)

# 开始训练
for epoch in range(MAX_EPOCHES):
    for i in range(MAX_EPISODES):
        s = env.reset()                 # 初始化回合设置
        for j in range(MAX_EP_STEPS):
            env.render()                # 环境的渲染
            a = rl.choose_action(s)     # RL 选择动作
            s_, r, done = env.step(a)   # 在环境中施加动作

            # DQN 这种强化学习需要存放记忆库
            rl.store_transition(s, a, r, s_)

            if rl.memory_full:
                rl.learn()              # 记忆库满了, 开始学习

            s = s_                      # 变为下一回合