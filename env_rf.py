# -*- coding: utf-8 -*-
"""
@Time          : 2021/3/10 10:54
@Author        : BarneyQ
@File          : env-rf.py
@Software      : PyCharm
@Description   : 微调RL环境
                                1. 环境读取一张图和Ground Truth坐标
                                2. 初始化环境在ground Truth 正中间（默认）
                                3. 初始框大小10×10
@Modification  :
    @Author    :
    @Time      :
    @Detail    :

"""
# -*- coding: utf-8 -*-
import pyglet
import numpy as np
import utilds_rf  as ut
import time
import torch
from  collections import deque
import random
GPU = 0
device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")



class MediaEnv(object):
    viewer = None
    action_dim = 6 # 6 个动作
    # state_dim = 4096+action_dim*ut.MAX_EP_STEPS # 4096+4096+700 vgg
    # state_dim = 4096+action_dim*ut.HIS_EPISODES # 4096+4096+700 vgg
    state_dim = 25088+action_dim*ut.HIS_EPISODES # 512+512+700 resnet -18 ： 512 + 30
    def __init__(self):
        self.img_box = [0,0,ut.IMG_WIDTH,ut.IMG_HEIGHT] #初始框
        self.state_info = {} # 存储环境信息
        self.state_info['img_box'] = self.img_box
        self.state_info['img_path'] = ut.IMG_PATH
        self.state_info['cur_iou'] = []

        self.feature_ext = ut.build_pretrain_model().to(device) # 预训练模型

        self.his_actions_deq = deque(maxlen=ut.HIS_EPISODES)  #历史动作deque
        self.his_actions_deq = ut.init_his_action_deq(self.his_actions_deq,
                                                      action_dim=self.action_dim)
        self.scale_img = [1.0,1.0] # 有缩放图片的用



    def step(self, action):
        done = False
        reset_f = False
        r = 0.
        self.img_box = self.state_info['img_box']
        #执行动作得到新的框
        self.img_box = ut.cmt_imgcor_by_action(list_imgcor=self.img_box,action=action)
       #去除异常坐标
        self.img_box = ut.drop_outliers(self.img_box)

        # self.state_info['img_box'] = self.img_box
        self.cur_iou = ut.cal_iou(self.state_info['img_box'],self.ground_box)
        # done and reward

        self.last_iou = self.state_info['cur_iou']
        delta_iou = self.cur_iou - self.last_iou

        if delta_iou > 0:
            r = 1.0
        elif delta_iou == 0:
            r = 0.0
        else:
            r = -1.0

        if action == 5:
            # 如果采取动作5，代表我认为停止
            if self.cur_iou >= ut.IOU_END:
                done = True
                r = 10.0
            else:
                # 否则，给惩罚
                r = -10.0



        self.state_info['cur_iou'] = self.cur_iou
        # next state : 1整图的  2ROI的  3动作
        # hol_img_feature,self.scale_img = ut.get_img_feature(self.feature_ext,
        #                              img_path=self.state_info['img_path'])
        # print(self.state_info['img_box'])
        roi_feature,self.scale_img = ut.get_img_feature(self.feature_ext,
                                         img_path=self.state_info['img_path'],
                                         roi=self.img_box)
        his_actions =ut.cmt_act_history(self.his_actions_deq,action_dim=self.action_dim,action=action)
        self.state_info['img_box'] = self.img_box

        # s_ = torch.cat((hol_img_feature,roi_feature,his_actions),dim=1)
        s_ = torch.cat((roi_feature,his_actions),dim=1)
        s_ = s_.cpu().detach().numpy()

        return s_,r,done,self.cur_iou



    def reset(self,index,files,labels):


        # hol_img_feature,self.scale_img = ut.get_img_feature(self.feature_ext,
        #                                      img_path=self.state_info['img_path'])

        roi_feature,self.scale_img = ut.get_img_feature(self.feature_ext,
                                         img_path=self.state_info['img_path'],
                                         roi=self.state_info['img_box'])

        self.ground_box = ut.get_ground_box(index, labels, self.scale_img)  # 获取标注框
        # 20210310 :初始框在ground truth的中心点 10 10 的矩形框上进行寻找
        # self.img_box = [(self.ground_box[0]+self.ground_box[2]/2-ut.INIT_BOX_WIDTH/2 + random.randint(0,11) ),
        #                             (self.ground_box[1]+self.ground_box[3]/2-ut.INIT_BOX_HEIGHT/2 + random.randint(0,11)),
        #                             ut.INIT_BOX_WIDTH, ut.INIT_BOX_HEIGHT]  # 初始框 [groud truth的中心点, 10，10]
        self.img_box = [0, 0, ut.IMG_WIDTH, ut.IMG_HEIGHT]  # 初始框 全图范围
        self.cur_iou = ut.cal_iou(self.img_box, self.ground_box)  # 获取当前iou

        self.state_info['img_box'] = self.img_box
        self.state_info['ground_box'] = self.ground_box
        self.state_info['cur_iou'] = self.cur_iou  # store iou
        self.state_info['img_path'] = files[index]

        self.init_iou = self.cur_iou  # 保留最初的iou 不能小于这个值

        his_actions = ut.cmt_act_history(self.his_actions_deq, action_dim=self.action_dim, action=None)
        # s_ = torch.cat((hol_img_feature, roi_feature, his_actions), dim=1)
        s_ = torch.cat(( roi_feature, his_actions), dim=1)
        s_ = s_.cpu().detach().numpy()
        return s_

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.state_info)
        self.viewer.render()

    def sample_action(self):
        action =  np.random.randint(5,13)    # 13 action
        return action



class Viewer(pyglet.window.Window):

    def __init__(self,state_info):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=ut.WIN_WIDTH, height=ut.WIN_HEIGHT, resizable=False,
                                     caption='MediaAI', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1) # backgound color

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)
        self.foreground0 = pyglet.graphics.OrderedGroup(2)

        # 读取信息
        self.state_info = state_info
        self.img_src = self.state_info['img_path']
        self.img = pyglet.image.load(self.img_src) # info1
        self.img_spr = pyglet.sprite.Sprite(self.img,batch=self.batch,group=self.background)

        self.img_box_cor = self.state_info['img_box']
        self.box_cor = ut.cvt_img2pyglet(self.img_box_cor)

        self.img_box_ground_cor = self.state_info['ground_box']
        self.box_ground_cor = ut.cvt_img2pyglet(self.img_box_ground_cor)
        self.img_box = self.batch.add(
            4, pyglet.gl.GL_LINE_LOOP, self.foreground0,  # 4 corners
            ('v2f', [self.box_cor[0], self.box_cor[1],  # x1, y1
                            self.box_cor[2], self.box_cor[3],  # x2, y2
                            self.box_cor[4], self.box_cor[5],  # x3, y3
                            self.box_cor[6], self.box_cor[7]]),  # x4, y4
            ('c3B', (255, 0, 0) * 4))  # color
        self.img_box_ground = self.batch.add(
            4, pyglet.gl.GL_LINE_LOOP, self.foreground,  # 4 corners
            ('v2f', [self.box_ground_cor[0], self.box_ground_cor[1],  # x1, y1
                            self.box_ground_cor[2], self.box_ground_cor[3],  # x2, y2
                            self.box_ground_cor[4], self.box_ground_cor[5],  # x3, y3
                            self.box_ground_cor[6], self.box_ground_cor[7]]),  # x4, y4
            ('c3B', (0, 255, 0) * 4))  # color
        pyglet.gl.glLineWidth(3)  # 设置线粗


    def render(self):
        self._update_info()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_info(self):
        self.img_src = self.state_info['img_path']
        self.img = pyglet.image.load(self.img_src)  # info1
        self.img_spr = pyglet.sprite.Sprite(self.img, batch=self.batch, group=self.background)

        self.img_box_cor = self.state_info['img_box']
        self.box_cor = ut.cvt_img2pyglet(self.img_box_cor)
        self.img_box.vertices = [self.box_cor[0], self.box_cor[1],  # x1, y1
                            self.box_cor[2], self.box_cor[3],  # x2, y2
                            self.box_cor[4], self.box_cor[5],  # x3, y3
                            self.box_cor[6], self.box_cor[7]]

        self.img_box_ground_cor = self.state_info['ground_box']
        self.box_ground_cor = ut.cvt_img2pyglet(self.img_box_ground_cor)
        self.img_box_ground.vertices = [self.box_ground_cor[0], self.box_ground_cor[1],  # x1, y1
                                 self.box_ground_cor[2], self.box_ground_cor[3],  # x2, y2
                                 self.box_ground_cor[4], self.box_ground_cor[5],  # x3, y3
                                 self.box_ground_cor[6], self.box_ground_cor[7]]


if __name__ == '__main__':
    env = MediaEnv()
    flag = 0
    env.render()
    while True:
        env.render()
        env.step(env.sample_action())
        # time.sleep(1)
