gpus = "0"
learning_rate = 1e-3
batch_size = 64
n_epochs = 30
model_name = "tf_efficientnet_b3_ns"
image_size = 384
fold = -1
drop_rate = 0.3
classes = None
num_classes = 137
smooth = 0.1
alpha = 0.4
long_resize = False
imbalance_sample = False
optim = "adamw"
offline = True
mask = False
proj = "baseline"
tag = "b3ns_drop0.3_mix0.4_imsz384_augv3_noclip"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import glob
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import timm
from torchsampler import ImbalancedDatasetSampler
from torch_optimizer import Yogi
from utils.loss.smooth import LabelSmoothingLoss
from utils.mixup import mixup_data, mixup_criterion
pl.seed_everything(0)

class Model(pl.LightningModule):
    def __init__(self, learning_rate = 1e-3, batch_size = 64, n_epochs = 30, model_name = "resnet18", image_size = 256, fold = 0, drop_rate = 0, num_classes = 137, smooth = 0, train_trans = None, valid_trans = None, criterion = None, alpha = 0, imbalance_sample = False, long_resize = False, optim = "adamw", mask = False, classes = None):
        super(Model, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_name = model_name
        self.image_size = image_size
        self.fold = fold
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.smooth = smooth
        self.train_trans = train_trans
        self.valid_trans = valid_trans
        self.criterion = criterion
        self.alpha = alpha
        self.imbalance_sample = imbalance_sample
        self.long_resize = long_resize
        self.optim = optim
        self.mask = mask
        self.classes = classes
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained = True, num_classes = num_classes, drop_rate = drop_rate)

    class Data(Dataset):
        def __init__(self, df, trans):
            self.df = df
            self.trans = trans

        def __getitem__(self, idx):
            label = self.df.loc[idx, "label"]
            file_name = self.df.loc[idx, "file_name"]
            image = np.array(Image.open(file_name).convert(mode = "RGB"))
            if self.trans is not None:
                image = self.trans(image = image)["image"]
            image = image.astype(np.float32).transpose(2, 0, 1)
            return image, label

        def __len__(self):
            return len(self.df)

    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.learning_rate, weight_decay = 2e-5)
        elif self.optim == "yogi":
            optimizer = Yogi(self.model.parameters(), lr = self.learning_rate)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.learning_rate, steps_per_epoch = int(len(self.train_dataloader())), epochs = self.n_epochs, anneal_strategy = "linear", final_div_factor = 30,), 'name': 'learning_rate', 'interval':'step', 'frequency': 1}
        return [optimizer], [lr_scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.alpha != 0:
            x, ya, yb, lam = mixup_data(x, y, self.alpha)
            yhat = self(x)
            loss = mixup_criterion(self.criterion, yhat, ya, yb, lam)
        else:
            yhat = self(x)
            loss = self.criterion(yhat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log("valid_loss", loss)
        return y, yhat

    def validation_step_end(self, output):
        return output

    def validation_epoch_end(self, outputs):
        y = torch.cat([_[0] for _ in outputs]).detach().cpu().numpy()
        yhat = torch.cat([_[1] for _ in outputs]).argmax(1).detach().cpu().numpy()
        acc = accuracy_score(y, yhat)
        self.log("valid_metric", acc, prog_bar = True)

    def setup(self, stage = None):
        if not self.mask:   
            image_files = glob.glob("./data/train/*/*.*")
        else:
            image_files = glob.glob("./data/masked_train/*/*.*")
        df = pd.DataFrame(image_files, columns = ["file_name"])
        df["label"] = df.file_name.apply(lambda x: int(x.split('/')[-2]))
        split = StratifiedKFold(5, shuffle = True, random_state = 0)
        train_idx, valid_idx = list(split.split(df, df.label))[self.fold]
        df_train = df.loc[train_idx].reset_index(drop = True) if self.fold != -1 else df.copy().reset_index(drop = True)
        df_valid = df.loc[valid_idx].reset_index(drop = True)
        if self.classes is not None:
            df_train = df_train[df_train.label.isin(self.classes)].reset_index(drop = True)
            df_valid = df_valid[df_valid.label.isin(self.classes)].reset_index(drop = True)
        self.ds_train = self.Data(df_train, self.train_trans)
        self.ds_valid = self.Data(df_valid, self.valid_trans)

    def train_dataloader(self):
        if self.imbalance_sample:
            sampler = ImbalancedDatasetSampler(self.ds_train, callback_get_label = lambda x: np.array(x.df.label))
            return DataLoader(self.ds_train, self.batch_size, sampler = sampler, num_workers = 4)
        else:
            return DataLoader(self.ds_train, self.batch_size, shuffle = True, num_workers = 4)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, self.batch_size, num_workers = 4)

if __name__ == "__main__":
    train_trans = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
        ], p = 0.9),
        A.OneOf([
            A.GridDistortion(),
            A.OpticalDistortion(),
        ], p = 0.9),
        A.Normalize()])
    valid_trans = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize()])
    logger = WandbLogger(project = proj, name = tag, version = tag + "_" + str(fold), offline = offline)
    callback = pl.callbacks.ModelCheckpoint(
        filename = '{epoch}_{valid_metric:.3f}',
        save_last = True,
        mode = "max",
        monitor = 'valid_metric'
    )
    model = Model(
        learning_rate = learning_rate, 
        batch_size = batch_size, 
        n_epochs = n_epochs, 
        model_name = model_name, 
        image_size = image_size, 
        fold = fold, 
        drop_rate = drop_rate, 
        num_classes = num_classes, 
        smooth = smooth, 
        train_trans = train_trans, 
        valid_trans = valid_trans,
        alpha = alpha,
        imbalance_sample = imbalance_sample,
        criterion = LabelSmoothingLoss(num_classes, smooth),
        long_resize = long_resize,
        optim = optim,
        mask = mask,
        classes = classes
    )
    trainer = pl.Trainer(
        gpus = len(gpus.split(",")), 
        precision = 16, amp_backend = "native", amp_level = "O1", 
        accelerator = "dp",
        gradient_clip_val = 0.5,
        max_epochs = n_epochs,
        stochastic_weight_avg = True,
        logger = logger,
        progress_bar_refresh_rate = 10,
        callbacks = [callback]
    )
    trainer.fit(model)
