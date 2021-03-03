# -*- coding: utf-8 -*-
"""
@Time          : 2021/3/1 16:30
@Author        : BarneyQ
@File          : environment.py
@Software      : PyCharm
@Description   :
    1. 实现动作
    2. 实现动作的奖励
    3. 可视化
@Modification  :
    @Author    :
    @Time      :
    @Detail    :

"""

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import pyglet




WIN_W = 720
WIN_H = 540
UNIT = 1
VIEW_H = 540
VIEW_W = 720
UNIT = 1  # 香酥鸡


class Env_Med(tk.Tk, object):
    Viewer = None
    def __init__(self):
        super(Env_Med, self).__init__()
        pass




    # reset
    def reset(self):
        pass
    # step
    def step(self,action):
        pass
    # render
    def render(self):
        if self.Viewer is None:
            self.Viewer = Viewer()
        self.Viewer.render()






class Viewer(pyglet.window.Window):
    def __init__(self):
        # 画出图像和检测框等
        super(Viewer,self).__init__(width=WIN_W, height=WIN_H, resizable=False, caption='MediaAI', vsync=False)
        # 窗口背景颜色
        pyglet.gl.glClearColor(1, 1, 1, 1)

        # 将手臂的作图信息放入这个 batch
        self.batch = pyglet.graphics.Batch()  # display whole batch at once
        background = pyglet.graphics.OrderedGroup(0)
        foreground = pyglet.graphics.OrderedGroup(1)


        # 添加蓝点
        img = pyglet.image.load('./data/caifenfang-0.png')
        # self.rec = pyglet.shapes.BorderedRectangle(100,100,200,200,border_color=(255,0,0),border=3,batch=self.batch,group=foreground)
        # # self.rec.opacity = 128
        self.img_sprite = pyglet.sprite.Sprite(img,batch=self.batch,group=background)
        # self.square.opacity = 255
        # 添加蓝点
        # self.rec =
        self.point = self.batch.add(
            4, pyglet.gl.GL_LINE_LOOP, foreground,  # 4 corners
            ('v2f', [100, 100,  # x1, y1
                     100, 200,  # x2, y2
                     300, 200,  # x3, y3
                     300, 100]),  # x4, y4
            ('c3B', (255, 0, 0) * 4))  # color
        pyglet.gl.glLineWidth(3) # 设置线粗



    def render(self):
        # 刷新并呈现在屏幕上
        self._update_state()  # 更新内容 (暂时没有变化)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()


    def on_draw(self):
        # 刷新图片和检测框位置
        self.clear()  # 清屏
        # self.img_sprite.draw()
        self.batch.draw()  # 画上 batch 里面的内容


    def _update_state(self):
        # 更新手臂的位置信息
        pass



########################### 测试环境 ###################################
if __name__ == '__main__':
    env = Env_Med()
    while True:
        env.render()

