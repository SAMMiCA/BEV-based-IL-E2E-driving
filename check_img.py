import cv2
from glob import glob
import numpy as np
import os

path = '/home/oem/jh/KAIC2022/dataset/run_20220922_164723/bb_bev'
imgs = glob(path + '/*.jpg')
imgs = sorted(imgs)
px_per_meter = 5

for index in range(len(imgs)-5*10):
    img = imgs[index]
    height, width, _ = cv2.imread(img).shape
    print(os.path.basename(img).split('_'))
    _, cur_x, cur_y, cur_yaw, _, _ = os.path.basename(img).split('_')
    img = cv2.imread(img)

    for i in range(1, 6):
        _, fut_x, fut_y, fut_yaw, _, _ = os.path.basename(imgs[index+i * 10]).split('_')

        cur_x = float(cur_x)
        cur_y = float(cur_y)
        cur_yaw = float(cur_yaw)

        fut_x = float(fut_x)
        fut_y = float(fut_y)

        mat1 = np.array([[1, 0, -cur_x],
                        [0, -1, +cur_y],
                        [0, 0, 1]])

        mat2 = np.array([[np.cos(np.pi/2-cur_yaw), np.sin(np.pi/2-cur_yaw), 0],
                        [-np.sin(np.pi/2-cur_yaw), np.cos(np.pi/2-cur_yaw), 0],
                        [0, 0, 1]])

        mat3 = np.array([[px_per_meter, 0, width/2],
                        [0, px_per_meter, height/2],
                        [0, 0, 1]])

        fut_vec = np.array([fut_x, fut_y, 1])

        fut_ego = mat3 @ (mat2 @ (mat1 @ fut_vec))

        print(fut_ego)

        img = cv2.circle(img, (int(fut_ego[0]), int(fut_ego[1])), 5, (0, 55 + 200//5 * i, 0), -1)
    cv2.imshow("img", img)

    # channel_sep = np.concatenate([img[...,0], img[...,1], img[...,2]], 1) 
    # cv2.imshow("seg", channel_sep)
    cv2.waitKey(0)