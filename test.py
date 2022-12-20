from utils import read_json
import os
import torch
from model.MyModel import MyModel
from dataset import MyDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
import tqdm

if __name__ == '__main__':
    logdir = 'logs/2022_09_27_03_17_00'
    CONFIG = read_json(os.path.join(logdir, "config.json"))
    device = torch.device("cuda:0")

    chkpt = torch.load(os.path.join(logdir, "model", "best_valid.pt"), map_location=device)
    model = MyModel(        
        CONFIG["IMG_SIZE"],
        CONFIG["HORIZON"],
        CONFIG["CHANNELS"],
        CONFIG["PATCH_SIZE"],
        CONFIG["DIM"],
        CONFIG["DEPTH"])

    model.load_state_dict(chkpt["model"])
    model.eval()
    model.to(device)

    dataset = MyDataset(
        # CONFIG["PATH"],
        "../dataset/",
        CONFIG["IMG_SIZE"],
        CONFIG["INTERVAL"],
        CONFIG["HORIZON"],
        CONFIG["PX_PER_METER"],
        CONFIG["BEV_NORM"],
        CONFIG["SPEED_NORM"],
        CONFIG["WAYPOINT_NORM_X"],
        CONFIG["WAYPOINT_NORM_Y"]
    )

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])


    bev_norm = np.array(CONFIG["BEV_NORM"])
    bev_norm = np.expand_dims(bev_norm, [0,1])

    test_dl = DataLoader(test_ds, 1, shuffle=False)
    errors = np.array([0]*10, dtype=np.float32)
    for batch in tqdm.tqdm(test_dl):
        (bev_batch, speed_batch, fut_ego_list) = batch
        bev = bev_batch[0].numpy().transpose(1,2,0)
        bev = bev * bev_norm[:,:,3:] + bev_norm[:,:,:3]
        bev = np.ascontiguousarray(bev).astype(np.uint8)

        pred = model(bev_batch.to(device), speed_batch.to(device))
        pred = torch.reshape(pred, (-1, CONFIG["HORIZON"], 2))[0]
        pred = pred.detach().cpu().numpy()
        sbev = np.stack([bev[:,:,0]]*3, 2)
        hdg = np.concatenate([bev[:,:, 1:], np.zeros_like(bev[:,:,0:1])], 2)
        for i, _ in enumerate(pred):
            pred[i][0] = (pred[i][0] * CONFIG["WAYPOINT_NORM_X"][i+CONFIG["HORIZON"]]) + CONFIG["WAYPOINT_NORM_X"][i]
            pred[i][1] = (pred[i][1] * CONFIG["WAYPOINT_NORM_Y"][i+CONFIG["HORIZON"]]) + CONFIG["WAYPOINT_NORM_Y"][i]
            sbev = cv2.circle(sbev, (int(pred[i][0]), int(pred[i][1])), 5, (0, 55 + 200//CONFIG["HORIZON"] * i, 0), -1)
            hdg = cv2.circle(hdg, (int(pred[i][0]), int(pred[i][1])), 5, (0, 55 + 200//CONFIG["HORIZON"] * i, 0), -1)
            bev = cv2.circle(bev, (int(pred[i][0]), int(pred[i][1])), 5, (0, 0, 55 + 200//CONFIG["HORIZON"] * i), -1)
            # print(pred_fut_ego)

        # cv2.imshow("corr", bev[:, :, 0])
        # cv2.imshow("bev", np.concatenate([bev[:,:,0], bev[:,:,1], bev[:,:,2]], 1))
        cv2.imshow("bev", bev)
        cv2.imshow("sbev", sbev)
        cv2.imshow("hdg", hdg)

        gt_bev = bev_batch[0].numpy().transpose(1,2,0)
        gt_bev = gt_bev * bev_norm[:,:,3:] + bev_norm[:,:,:3]
        gt_bev = np.ascontiguousarray(gt_bev).astype(np.uint8)
        
        fut_ego_list = torch.reshape(fut_ego_list, (-1, CONFIG["HORIZON"], 2))[0]
        fut_ego_list = fut_ego_list.numpy()
        for i, _ in enumerate(fut_ego_list):
            fut_ego_list[i][0] = (fut_ego_list[i][0] * CONFIG["WAYPOINT_NORM_X"][i+CONFIG["HORIZON"]]) + CONFIG["WAYPOINT_NORM_X"][i]
            fut_ego_list[i][1] = (fut_ego_list[i][1] * CONFIG["WAYPOINT_NORM_Y"][i+CONFIG["HORIZON"]]) + CONFIG["WAYPOINT_NORM_Y"][i]
            gt_bev = cv2.circle(gt_bev, (int(fut_ego_list[i][0]), int(fut_ego_list[i][1])), 5, (0, 0, 55 + 200//CONFIG["HORIZON"] * i), -1)

        
        # print( np.sum((pred - fut_ego_list)**2,1 ))
        errors += np.sum((pred - fut_ego_list)**2,1 )
        print(errors)

        # error = pred - fut_ego_list

        # cv2.imshow("gt_bev", gt_bev)
        key = cv2.waitKey(0)
        if key == ord("q"): break
        else: pass
    print(np.sqrt(errors/test_size)/5)
