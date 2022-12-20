from ssl import CHANNEL_BINDING_TYPES
from dataset import MyDataset
from model.MyModel import MyModel
import torch
import datetime
import os
from utils import save_json
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm
import torch.nn.functional as F
import wandb

CONFIG = dict(
    MODEL_NAME          = "mlp-mixer-bbox-LN",
    INPUT               = ["bbox+hdmap+gp", "speed"],
    BATCH_SIZE          = 64,
    NUM_EPOCHS          = 1000,
    LEARNING_RATE       = 1e-4,
    PATIENCE            = 3,

    CHANNELS            = 3,
    PATCH_SIZE          = 20,
    DIM                 = 256,
    DEPTH               = 8,

    PATH                = "/home/oem/jh/KAIC2022/dataset",
    IMG_SIZE            = (600, 300),
    INTERVAL            = 10,
    HORIZON             = 10,
    PX_PER_METER        = 5,
    BEV_NORM            = [49.7, 91.2, 9.4, 69.9, 115.3, 44.1],
    SPEED_NORM          = [4.24, 2.50],
    WAYPOINT_NORM_X     = [150., 149.6, 148.6, 147.0, 145.1, 142.7, 139.9, 136.8, 133.4, 129.8,
                           0., 1.6,  5.2, 10.3, 16.6, 23.7, 31.7, 40.3,  49.3, 59.0],
    WAYPOINT_NORM_Y     = [300., 282.8, 265.7, 249.0, 232.7, 216.9, 201.6, 186.8, 172.6,  159.0,
                           0.  , 10.1, 19.7, 28.7, 37.2, 45.4, 53.1, 60.4, 67.4,  74.2],
)

log_wandb = True

if __name__ == "__main__":

    if log_wandb:
        wandb.init(project="KAIC2022", entity="jhoonauto", name=f'IL_v1')
        wandb.config = CONFIG

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    logdir = os.path.join("logs", timestamp)
    os.makedirs(logdir)
    os.makedirs(os.path.join(logdir, "log"))
    os.makedirs(os.path.join(logdir, "model"))

    ## save the config
    save_json(CONFIG, os.path.join(logdir, "config.json"))

    model = MyModel(        
        CONFIG["IMG_SIZE"],
        CONFIG["HORIZON"],
        CONFIG["CHANNELS"],
        CONFIG["PATCH_SIZE"],
        CONFIG["DIM"],
        CONFIG["DEPTH"])


    model.to(device)

    dataset = MyDataset(
        CONFIG["PATH"],
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
    valid_size = len(dataset) - train_size

    torch.manual_seed(42)
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_dl = DataLoader(train_ds, CONFIG["BATCH_SIZE"], shuffle=True)
    valid_dl = DataLoader(valid_ds, CONFIG["BATCH_SIZE"], shuffle=False)

    optimizer = optim.Adam(model.parameters(), CONFIG["LEARNING_RATE"])

    best_train = float("Inf")
    best_valid = float("Inf")
    es_trigger = 0

    for epoch in tqdm.tqdm(range(CONFIG["NUM_EPOCHS"]), "Epoch", position=1):
        epoch_train_loss = 0
        epoch_valid_loss = 0

        model.train()
        for batch in tqdm.tqdm(train_dl, "Training Batch", position=0, leave=False):

            bev_batch, speed_batch, fut_ego_batch = batch
            cbs = int(bev_batch.shape[0])
            bev_batch, speed_batch, fut_ego_batch = bev_batch.to(device), speed_batch.to(device), fut_ego_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(bev_batch, speed_batch)
            pred = torch.reshape(pred, (-1, CONFIG["HORIZON"], 2))
            loss = F.mse_loss(pred, fut_ego_batch)
            epoch_train_loss += loss.item()

            if log_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': loss.item()
                })

            loss = torch.sqrt(loss)
            loss.backward()
            optimizer.step()

            epoch_train_loss += (loss.item() * cbs)

        model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(valid_dl, "Validation Batch", position=0, leave=False):

                bev_batch, speed_batch, fut_ego_batch = batch
                cbs = int(bev_batch.shape[0])
                bev_batch, speed_batch, fut_ego_batch = bev_batch.to(device), speed_batch.to(device), fut_ego_batch.to(device)
                pred = model(bev_batch, speed_batch)
                pred = torch.reshape(pred, (-1, CONFIG["HORIZON"], 2))
                loss = F.mse_loss(pred, fut_ego_batch)
                epoch_valid_loss += loss.item()

                if log_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'valid_loss': loss.item()
                    })

                epoch_valid_loss += (loss.item()) * cbs

        epoch_train_loss = epoch_train_loss / train_size
        epoch_valid_loss = epoch_valid_loss / valid_size

        tqdm.tqdm.write("Training loss: %.4f || Validation loss: %.4f " 
                         % (epoch_train_loss, epoch_valid_loss))

        chkpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": epoch_train_loss,
            "valid_loss": epoch_valid_loss,
            "epoch": epoch
        }

        torch.save(chkpt, os.path.join(logdir, "model", "latest.pt"))

        if epoch_train_loss <= best_train:
            best_train = epoch_train_loss
            torch.save(chkpt, os.path.join(logdir, "model", "best_train.pt"))

        if epoch_valid_loss <= best_valid:
            best_valid = epoch_valid_loss
            torch.save(chkpt, os.path.join(logdir, "model", "best_valid.pt"))
        
        else:
            es_trigger += 1
            print("Early stopping triggers: %d" % es_trigger)

            if es_trigger >= CONFIG["PATIENCE"]:
                print("Early stopping the experiment!")
                break
