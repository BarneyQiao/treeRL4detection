# -*- coding: utf-8 -*-
"""
@Time          : 2021/3/10 14:01
@Author        : BarneyQ
@File          : utilds-rf.py
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
import os
GPU = 0
device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

# Viewer and env
WIN_WIDTH = 720
WIN_HEIGHT = 540
IMG_WIDTH = 720
IMG_HEIGHT = 540
IMG_PATH = './assets/caifenfang-0.png'

SCALE_FACTOR = 0.80 # 执行一次动作，变为0.75倍
TRANS_FACTOR = 0.80
IOU_END = 0.6  #最低IOU

INIT_BOX_WIDTH = 500  #初始框的宽
INIT_BOX_HEIGHT = 100 # 初始框的高

# train and  test
MAX_EPOCHES = 30 # 训练epoches
MAX_EPISODES = 1 # 训练集图片个数
MAX_EP_STEPS = 10  #每个图片的交互步数
EPS = 1.0
HIS_EPISODES = 5   # 历史动作记住


#Agent
BATCH_SIZE = 100
LEARNING_RATE = 1e-6
GAMMA = 0.9     # reward discount

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
# image
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
])

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
def cmt_imgcor_by_action_old(list_imgcor,action):
    x = list_imgcor[0]
    y = list_imgcor[1]
    w = list_imgcor[2]
    h = list_imgcor[3]
    # scale 从小往大，不需要scale变小
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
    # if action ==0:
    #     # →
    #     x = x+w * TRANS_FACTOR
    # elif action == 1:
    #     # ←
    #     x = x-w*TRANS_FACTOR
    # elif action == 2:
    #     # ↓
    #     y = y+h*TRANS_FACTOR
    # elif action == 3:
    #     # ↑
    #     y = y-h*TRANS_FACTOR
    # elif action == 4:
    #     # 上下压扁
    #     x_t = x + w / 2
    #     y_t = y + h / 2
    #     h = h * (1-TRANS_FACTOR *2)
    #     y = y_t- h/2
    # elif action == 5:
    #     # 左右压瘦
    #     x_t = x + w / 2
    #     y_t = y + h / 2
    #     w = w * (1-TRANS_FACTOR *2)
    #     x = x_t -  w/2
    # elif action == 6:
    #     # 上下拉长
    #     x_t = x + w / 2
    #     y_t = y + h / 2
    #     h = h * (1 +  TRANS_FACTOR *2)
    #     y = y_t - h/2
    # elif action == 7:
    #     # 左右拉胖
    #     x_t = x + w / 2
    #     y_t = y + h / 2
    #     w = w * (1 +  TRANS_FACTOR * 2)
    #     x = x_t - w/2
    # # 加上两个scale的动作 中心变小 和中心变大
    # elif action == 8:
    # # 中间 缩小 scale
    # #     x_t = x + w/2
    # #     y_t = y + h/2
    #     w = w - w * SCALE_FACTOR
    #     h = h - h * SCALE_FACTOR
    #     # x = x_t-w/2
    #     # y = y_t-h/2
    # elif action == 9:
    #     # 中间扩大 scale
    #     # x_t = x + w / 2
    #     # y_t = y + h / 2
    #     w = w + w * SCALE_FACTOR
    #     h = h + h * SCALE_FACTOR
    #     # x = x_t + w / 2
    #     # y = y_t + h / 2
    else:
        # stop动作
        pass
    return [x,y,w,h]

# 执行动作之后的坐标(img) [x,y,w,h]
def cmt_imgcor_by_action(list_imgcor,action):
    x = list_imgcor[0]
    y = list_imgcor[1]
    w = list_imgcor[2]
    h = list_imgcor[3]
    # scale 从小往大，不需要scale变小
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
    # if action ==0:
    #     # →
    #     x = x+w * TRANS_FACTOR
    # elif action == 1:
    #     # ←
    #     x = x-w*TRANS_FACTOR
    # elif action == 2:
    #     # ↓
    #     y = y+h*TRANS_FACTOR
    # elif action == 3:
    #     # ↑
    #     y = y-h*TRANS_FACTOR
    # elif action == 4:
    #     # 上下压扁
    #     x_t = x + w / 2
    #     y_t = y + h / 2
    #     h = h * (1-TRANS_FACTOR *2)
    #     y = y_t- h/2
    # elif action == 5:
    #     # 左右压瘦
    #     x_t = x + w / 2
    #     y_t = y + h / 2
    #     w = w * (1-TRANS_FACTOR *2)
    #     x = x_t -  w/2
    # elif action == 6:
    #     # 上下拉长
    #     x_t = x + w / 2
    #     y_t = y + h / 2
    #     h = h * (1 +  TRANS_FACTOR *2)
    #     y = y_t - h/2
    # elif action == 7:
    #     # 左右拉胖
    #     x_t = x + w / 2
    #     y_t = y + h / 2
    #     w = w * (1 +  TRANS_FACTOR * 2)
    #     x = x_t - w/2
    # # 加上两个scale的动作 中心变小 和中心变大
    # elif action == 8:
    # # 中间 缩小 scale
    # #     x_t = x + w/2
    # #     y_t = y + h/2
    #     w = w - w * SCALE_FACTOR
    #     h = h - h * SCALE_FACTOR
    #     # x = x_t-w/2
    #     # y = y_t-h/2
    # elif action == 9:
    #     # 中间扩大 scale
    #     # x_t = x + w / 2
    #     # y_t = y + h / 2
    #     w = w + w * SCALE_FACTOR
    #     h = h + h * SCALE_FACTOR
    #     # x = x_t + w / 2
    #     # y = y_t + h / 2
    else:
        # stop动作
        pass
    return [x,y,w,h]



# 返回一个pre-trained模型vgg
def build_pretrain_model():
    # 加载模型
    # model = trained_models.vgg16_bn(pretrained= False)
    # features = model.features
    # vgg = VGG(features)
    # vgg.load_state_dict(torch.load('./vgg16_bn-6c64b313.pth'),strict= False)
    # vgg.eval()
    # return vgg
    model = trained_models.resnet18(pretrained=False)
    rs = Resnet18(model)
    rs.load_state_dict(torch.load('./rs-18.pth'), strict=False)
    print('load res18 parameters')
    rs.eval()


    return rs

# 给一个图片，使用预训练的VGG-16进行特征提取，最后输出一个4096维度的vector
def get_img_feature(model,img_path,roi=None):
    img = cv2.imread(img_path)
    img_org_w = img.shape[1]
    img_org_h = img.shape[0]

    scale_w = img_org_w/IMG_WIDTH
    scale_h = img_org_h/IMG_HEIGHT


    img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))

    if roi is not None:
        roi_x = int(roi[0])
        roi_y = int(roi[1])
        roi_w = int(roi[2])
        roi_h = int(roi[3])
        roi_y1 = roi_y+roi_h
        roi_x1 = roi_x+roi_w


        img = img[roi_y:roi_y1,roi_x:roi_x1]

    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor,0)
    feature =  model(img_tensor.to(device))
    return feature,[scale_w,scale_h]

#初始化一个动作历史deque避免不足的情况
def init_his_action_deq(his_actions_deq,action_dim):
    for i in range(HIS_EPISODES):
        his_actions_deq.append(np.zeros(action_dim))
    return his_actions_deq

# 拼凑一个650-d的历史动作集合,输入deque、当前动作、动作维度
def cmt_act_history(his_actions_deq,action,action_dim):
    if action is not None:
        action_vector = np.zeros(action_dim) # 变成13维度的vector
        action_vector[action] = 1
        his_actions_deq.append(action_vector)
    res = []
    for i in range(HIS_EPISODES):
        res.append(his_actions_deq[i])
    res = np.array(res)
    r = np.concatenate(res,axis=0)
    r = torch.from_numpy(r)
    r = torch.unsqueeze(r,0).to(device)
    return r

#去掉异常值
def drop_outliers(img_box):
    x1 = img_box[0]
    y1 = img_box[1]
    x2 = x1 + img_box[2]
    y2 = y1+ img_box[3]

    img_box[0] = np.clip(x1,2,IMG_WIDTH-2)
    x2 = np.clip(x2,x1,IMG_WIDTH-1)
    img_box[1] = np.clip(y1,2,IMG_HEIGHT-2)
    y2 = np.clip(y2,y1,IMG_HEIGHT-2)
    img_box[2] = x2 - img_box[0]
    img_box[2] = np.clip(img_box[2],2,IMG_WIDTH-2)
    img_box[3] = y2 - img_box[1]
    img_box[3] = np.clip(img_box[3],2,IMG_HEIGHT-2)

    return img_box

#计算IOU
def cal_iou(bbx, bbx_gt):
    x1 = bbx[0]
    y1 = bbx[1]
    x2 = x1+bbx[2]
    y2 = y1+bbx[3]

    X1 = bbx_gt[0]
    Y1 = bbx_gt[1]
    X2 = X1 + bbx_gt[2]
    Y2 = Y1 + bbx_gt[3]

    s1 = (y2-y1)*(x2-x1)
    s2 = (Y2-Y1)*(X2-X1)
    jx1 = max(x1,X1)
    jx2 = min(x2,X2)
    jy1 = max(y1,Y1)
    jy2 = min(y2,Y2)
    if jx2>jx1 and jy2>jy1:
        s3 = (jx2-jx1)*(jy2-jy1)
    else:
        s3 = 0.0
    return s3/(s1+s2-s3)

#返回本图片的ground_truth
def get_ground_box(index,labels,scale):

    ground_box = labels[index]
    ground_box[0] /= scale[0]
    ground_box[1] /= scale[1]
    return ground_box


#获取图片路径+标注框
def get_img_boxes():
    files=[]
    file_name = './assets-1/'
    labels = []
    for file in os.listdir('./assets-1'):
        tmp = file_name+file
        files.append(tmp)

    f = open('./tumorxy_label.txt')
    lines = f.readlines()

    for line in lines:
        str_1 = line.split(' ')
        str_2 = str_1[1]
        str_3 = str_2.split(',')
        x1 = int(str_3[0])
        y1 = int(str_3[1])
        x2 = int(str_3[2])
        y2 = int(str_3[3])
        label = [x1,y1,(x2-x1),(y2-y1)]
        labels.append(label)

    return files,labels


# 用于特征提取的VGG 输入224*224*3  输出4096个特征
class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Resnet18(nn.Module):
    def __init__(self,model):
        super(Resnet18, self).__init__()
        self.features = model

    def forward(self,x):
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        x = self.features.layer1(x)
        x = self.features.layer2(x)
        x = self.features.layer3(x)
        x = self.features.layer4(x)

        # x = self.features.avgpool(x)
        x = torch.flatten(x, 1)
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
    # vgg = build_pretrain_model()
    # feature = get_img_feature(vgg,'./assets/caifenfang-0.png',roi=[0,0,100,100])
    # print(feature.shape)
    files,labels = get_img_boxes()
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        cv2.imwrite(img=img,filename=file)
        print('done:{0}'.format(file))

    import torchvision.models as md
    from  torchvision.ops import RoIPool
    farnn = md.detection.FasterRCNN()
    roipool = RoIPool((7,7),1/16)

    # pool = roipool(x, indices_and_rois)