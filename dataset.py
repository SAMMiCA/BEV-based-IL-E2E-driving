import torch

from glob import glob
import os
import numpy as np
import cv2
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import tqdm


class ScenarioDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size, interval, horizon, px_per_meter, bev_norm, speed_norm, waypoint_norm_x, waypoint_norm_y):
        self.bb_bev_paths = sorted(glob(path+'/bb_bev/*.jpg'))
        self.gp_bev_paths = sorted(glob(path+'/gp_bev/*.jpg'))
        self.hd_bev_paths = sorted(glob(path+'/hd_bev/*.jpg'))
        assert len(self.bb_bev_paths)==len(self.gp_bev_paths)==len(self.hd_bev_paths)
        assert interval>=1
        assert horizon>=2
        self.interval = interval
        self.horizon = horizon
        self.px_per_meter = px_per_meter
        self.height, self.width = img_size
        self.bev_norm = np.array(bev_norm)
        self.bev_norm = np.expand_dims(self.bev_norm, [0,1])
        self.speed_norm = speed_norm
        self.waypoint_norm_x, self.waypoint_norm_y = np.array(waypoint_norm_x), np.array(waypoint_norm_y)

    def __len__(self):
        term = (self.horizon-1) * self.interval + 1
        return len(self.bb_bev_paths) - term + 1

    def __getitem__(self, idx):
        bb_bev_path = self.bb_bev_paths[idx]
        gp_bev_path = self.gp_bev_paths[idx]
        hd_bev_path = self.hd_bev_paths[idx]

        bb_bev = (cv2.imread(bb_bev_path)[:, :, 0]).astype(np.uint8)
        gp_bev = cv2.imread(gp_bev_path)[:, :, 0]
        hd_bev = cv2.imread(hd_bev_path)[:, :, 0]

        cv2.rectangle(bb_bev, (self.width//2-5, self.height//2-12), (self.width//2+4, self.height//2+11), 255, -1)

        bev = np.stack([bb_bev,
                        hd_bev,
                        gp_bev], -1)
        bev = (bev - self.bev_norm[:,:,:3]) / self.bev_norm[:,:,3:]
        bev = np.transpose(bev, (2, 0, 1))

        filename = os.path.splitext(os.path.basename(bb_bev_path))[0]
        file_id, cur_x, cur_y, cur_yaw, speed,flag = filename.split('_')
        cur_x = float(cur_x)
        cur_y = float(cur_y)
        cur_yaw = float(cur_yaw)
        speed = float(speed)
        speed= (speed - self.speed_norm[0]) / self.speed_norm[1]
        
        mat1 = np.array([[1, 0, -cur_x],
                        [0, -1, +cur_y],
                        [0, 0, 1]])
        mat2 = np.array([[np.cos(np.pi/2-cur_yaw), np.sin(np.pi/2-cur_yaw), 0],
                        [-np.sin(np.pi/2-cur_yaw), np.cos(np.pi/2-cur_yaw), 0],
                        [0, 0, 1]])
        mat3 = np.array([[self.px_per_meter, 0, self.width/2],
                        [0, self.px_per_meter, self.height/2],
                        [0, 0, 1]])
        mat = mat3 @ mat2 @ mat1
        

        fut_ego_list = []
        for i in range(self.horizon):
            _idx = idx + i*self.interval
            _cam_bev_path = self.bb_bev_paths[_idx]
            _filename = os.path.splitext(os.path.basename(_cam_bev_path))[0]
            _, fut_x, fut_y, fut_yaw, _, _ = _filename.split('_')
            fut_x = float(fut_x)
            fut_y = float(fut_y)
            fut_yaw = float(fut_yaw)

            fut_vec = np.array([fut_x, fut_y, 1])
            fut_ego = mat @ fut_vec
            fut_ego_list.append(fut_ego[:2])
        
        fut_ego_list = np.array(fut_ego_list, dtype=np.float32)

        fut_ego_list[:,0] = (fut_ego_list[:,0]-self.waypoint_norm_x[:self.horizon])/(self.waypoint_norm_x[self.horizon:]+1e-6)
        fut_ego_list[:,1] = (fut_ego_list[:,1]-self.waypoint_norm_y[:self.horizon])/(self.waypoint_norm_y[self.horizon:]+1e-6)


        return bev.astype(np.float32), np.array([speed], dtype=np.float32), fut_ego_list


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size, interval, horizon, px_per_meter, bev_norm, speed_norm, waypoint_norm_x, waypoint_norm_y):
        
        scenario_folders = glob(path + '/*/')
        datasets = []
        for scenario_folder in scenario_folders:
            datasets.append(ScenarioDataset(scenario_folder, img_size, interval, horizon, px_per_meter, bev_norm, speed_norm, waypoint_norm_x, waypoint_norm_y))
        self.dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.dataset)
        # return 20

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == '__main__':
    # mydataset = MyDataset('dataset', 0, 0)
    interval = 10
    horizon = 10
    px_per_meter = 5
    img_size = (600,300)
    bev_norm = [49.7, 91.2, 9.4, 69.9, 115.3, 44.1]
    speed_norm = [4.24, 2.50]
    waypoint_norm_x = [150., 149.6, 148.6, 147.0, 145.1, 142.7, 139.9, 136.8, 133.4, 129.8,
                           0., 1.6,  5.2, 10.3, 16.6, 23.7, 31.7, 40.3,  49.3, 59.0]
    waypoint_norm_y = [300., 282.8, 265.7, 249.0, 232.7, 216.9, 201.6, 186.8, 172.6,  159.0,
                           0.  , 10.1, 19.7, 28.7, 37.2, 45.4, 53.1, 60.4, 67.4,  74.2]
    mydataset = MyDataset('../dataset/', img_size, interval, horizon, px_per_meter, 
                          bev_norm, 
                          speed_norm,
                          waypoint_norm_x,
                          waypoint_norm_y)
    dataloader = DataLoader(mydataset, batch_size=3, shuffle=False)
    
    # bev_r_mean_list = []
    # bev_g_mean_list = []
    # bev_b_mean_list = []

    # bev_r_std_list = []
    # bev_g_std_list = []
    # bev_b_std_list = []

    # speed_list = []

    # fut_ego_x_list = []
    # fut_ego_y_list = []

    # np.set_printoptions(threshold=11)

    # for data in tqdm.tqdm(mydataset):
    #     bev, speed, fut_ego_list = data

    #     bev_r_mean_list.append(np.mean(bev[0]))
    #     bev_g_mean_list.append(np.mean(bev[1]))
    #     bev_b_mean_list.append(np.mean(bev[2]))

    #     bev_r_std_list.append(np.std(bev[0]))
    #     bev_g_std_list.append(np.std(bev[1]))
    #     bev_b_std_list.append(np.std(bev[2]))

    #     speed_list.append(speed)

    #     fut_ego_x_list.append(fut_ego_list[:,0])
    #     fut_ego_y_list.append(fut_ego_list[:,1])

    #     print(np.mean(bev_r_mean_list), np.mean(bev_r_std_list))
    #     print(np.mean(bev_g_mean_list), np.mean(bev_g_std_list))
    #     print(np.mean(bev_b_mean_list), np.mean(bev_b_std_list))

    #     print(np.mean(speed_list), np.std(speed_list))
    #     print(np.mean(np.array(fut_ego_x_list), 0), np.std(np.array(fut_ego_x_list), 0))
    #     print(np.mean(np.array(fut_ego_y_list), 0), np.std(np.array(fut_ego_y_list), 0))

    #     print('-----------------------')


    bev_norm = np.array(bev_norm)
    bev_norm = np.expand_dims(bev_norm, [0,1])
    

    for batch in dataloader:
        (bev, speed, fut_ego_list) = batch
        bev = bev[0].numpy().transpose(1,2,0)
        bev = bev * bev_norm[:,:,3:] + bev_norm[:,:,:3]
        bev = np.ascontiguousarray(bev).astype(np.uint8)
        
        bev = np.stack([bev[:,:,1], bev[:,:,2], bev[:,:,0]], -1)


        fut_ego_list = fut_ego_list[0]
        canvas = np.zeros((img_size[0],img_size[1],3))
        for i, fut_ego in enumerate(fut_ego_list):
            fut_ego[0] = (fut_ego[0] * waypoint_norm_x[i+horizon]) + waypoint_norm_x[i]
            fut_ego[1] = (fut_ego[1] * waypoint_norm_y[i+horizon]) + waypoint_norm_y[i]
            # print(fut_ego)
            # print(np.mean(bev[:,:,0]), np.std(bev[:,:,0]))
            # print(np.mean(bev[:,:,1]), np.std(bev[:,:,1]))
            # print(np.mean(bev[:,:,2]), np.std(bev[:,:,2]))
            # print("----")
            # bev = cv2.circle(bev, (int(fut_ego[0]), int(fut_ego[1])), 5, (0, 55 + 200//horizon * i, 0), -1)

        # print(speed[0])
        cv2.imshow("bev_0", bev[:,:,0])
        cv2.imshow("bev_1", bev[:,:,1])
        cv2.imshow("bev_2", bev[:,:,2])
        cv2.waitKey(0)