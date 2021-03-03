# -*- coding: utf-8 -*-
"""
@Time          : 2021/3/3 9:19
@Author        : BarneyQ
@File          : utils.py
@Software      : PyCharm
@Description   :
@Modification  :
    @Author    :
    @Time      :
    @Detail    :

"""
import torch
import torch.nn as nn
import torchvision.models as trained_models
import numpy as np
from collections import deque
import torchvision.transforms as transforms
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Viewer
WIN_WIDTH = 720
WIN_HEIGHT = 540
IMG_WIDTH = 720
IMG_HEIGHT = 540
IMG_PATH = './assets/caifenfang-0.png'

SCALE_FACTOR = 0.75
TRANS_FACTOR = 0.25


# train and  test
MAX_EPOCHES = 25 # 训练epoches
MAX_EPISODES = 10 # 训练集图片个数
MAX_EP_STEPS = 50  #每个图片的交互步数

# image
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(), ])

#给一个图片坐标，转换成glet坐标
def cvt_img2pyglet(list_img_cor):
    x = list_img_cor[0]
    y = list_img_cor[1]
    w = list_img_cor[2]
    h = list_img_cor[3]

    x1 = x
    y1 = IMG_HEIGHT - (y+h)
    x2 = x
    y2 = IMG_HEIGHT - y
    x3 = x +w
    y3 = IMG_HEIGHT - y
    x4 = x + w
    y4 = IMG_HEIGHT - (y + h)

    return [x1,y1,x2,y2,x3,y3,x4,y4]

# 执行动作之后的坐标(img) [x,y,w,h]
def cmt_imgcor_by_action(list_imgcor,action):
    x = list_imgcor[0]
    y = list_imgcor[1]
    w = list_imgcor[2]
    h = list_imgcor[3]
    # scale
    if action == 0:
        #左上
        w = w * SCALE_FACTOR
        h = h * SCALE_FACTOR
    elif action ==1:
        #右上
        x = x+ w*(1-SCALE_FACTOR)
        w = w * SCALE_FACTOR
        h = h * SCALE_FACTOR
    elif action ==2:
        #左下
        y = y + h * (1 - SCALE_FACTOR)
        w = w * SCALE_FACTOR
        h = h * SCALE_FACTOR
    elif action ==3:
        #右下
        x = x + w * (1 - SCALE_FACTOR)
        y = y + h * (1 - SCALE_FACTOR)
        w = w * SCALE_FACTOR
        h = h * SCALE_FACTOR
    elif action ==4:
        # 中间
        x_t = x + w/2
        y_t = y + h/2
        w = w * SCALE_FACTOR
        h = h * SCALE_FACTOR
        x = x_t-w/2
        y = y_t-h/2

    # trans
    elif action ==5:
        x = x+w * TRANS_FACTOR
    elif action == 6:
        x = x-w*TRANS_FACTOR
    elif action == 7:
        y = y+h*TRANS_FACTOR
    elif action == 8:
        y = y-h*TRANS_FACTOR
    elif action == 9:
        x_t = x + w / 2
        y_t = y + h / 2
        h = h * (1-2*TRANS_FACTOR)
        y = y_t- h/2
    elif action == 10:
        x_t = x + w / 2
        y_t = y + h / 2
        w = w * (1-2*TRANS_FACTOR)
        x = x_t -  w/2
    elif action == 11:
        x_t = x + w / 2
        y_t = y + h / 2
        h = h * (1 + 2 * TRANS_FACTOR)
        y = y_t - h/2
    elif action == 12:
        x_t = x + w / 2
        y_t = y + h / 2
        w = w * (1 + 2 * TRANS_FACTOR)
        x = x_t - w/2
    else:
        pass
    return [x,y,w,h]

# 返回一个pre-trained模型vgg
def build_pretrain_model():
    # 加载模型
    model = trained_models.vgg16_bn(pretrained= False)
    features = model.features
    vgg = VGG(features)
    vgg.load_state_dict(torch.load('./vgg16_bn-6c64b313.pth'),strict= False)
    vgg.eval()
    return vgg

# 给一个图片，使用预训练的VGG-16进行特征提取，最后输出一个4096维度的vector
def get_img_feature(model,img_path,roi=None):
    img = cv2.imread(img_path)

    if roi is not None:
        roi_x = int(roi[0])
        roi_y = int(roi[1])
        roi_w = int(roi[2])
        roi_h = int(roi[3])
        img = img[roi_y:roi_y+roi_h,roi_x:roi_w]

    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor,0)
    feature =  model(img_tensor.to(device))
    return feature

#初始化一个动作历史deque避免不足的情况
def init_his_action_deq(his_actions_deq,action_dim):
    for i in range(MAX_EP_STEPS):
        his_actions_deq.append(np.zeros(action_dim))
    return his_actions_deq

# 拼凑一个650-d的历史动作集合,输入deque、当前动作、动作维度
def cmt_act_history(his_actions_deq,action,action_dim):
    if action is not None:
        action_vector = np.zeros(action_dim) # 变成13维度的vector
        action_vector[action] = 1
        his_actions_deq.append(action_vector)
    res = []
    for i in range(MAX_EP_STEPS):
        res.append(his_actions_deq[i])
    res = np.array(res)
    r = np.concatenate(res,axis=0)
    r = torch.from_numpy(r)
    r = torch.unsqueeze(r,0).to(device)
    return r


def drop_outliers(img_box):
    x1 = img_box[0]
    y1 = img_box[1]
    x2 = x1 + img_box[2]
    y2 = y1+ img_box[3]

    img_box[0] = np.clip(x1,0,IMG_WIDTH-1)
    x2 = np.clip(x2,x1,IMG_WIDTH-1)
    img_box[1] = np.clip(y1,0,IMG_HEIGHT-1)
    y2 = np.clip(y2,y1,IMG_HEIGHT-1)
    img_box[2] = x2 - img_box[0]
    img_box[3] = y2 - img_box[1]

    return img_box



# 用于特征提取的VGG 输入224*224*3  输出4096个特征
class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



if __name__ == '__main__':
    # vgg = build_pretrain_model()
    # print(vgg)
    # a = deque(maxlen=50)
    # a = init_his_action_deq(a,13)
    # b = cmt_act_history(a,1,13)
    #
    # print(a)
    # print(b)
    vgg = build_pretrain_model()
    feature = get_img_feature(vgg,'./assets/caifenfang-0.png',roi=[0,0,100,100])
    print(feature.shape)