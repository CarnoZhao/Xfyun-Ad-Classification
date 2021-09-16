gpus = "2,3"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
import warnings
warnings.filterwarnings("ignore")

import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import timm
from utils.loss.smooth import LabelSmoothingLoss, SoftCrossEntropyLoss
from utils.mixup import mixup_data, mixup_criterion
pl.seed_everything(0)


class Model(pl.LightningModule):
    def __init__(self, **args):
        super(Model, self).__init__()
        for k, v in args.items():
            setattr(self, k, v)
        if self.classes is not None:
            self.num_classes = self.classes
        self.args = args
        self.model = timm.create_model(self.model_name, pretrained = True, num_classes = self.num_classes, drop_rate = self.drop_rate)
        self.criterion = SoftCrossEntropyLoss()
        self.save_hyperparameters()

    class Data(Dataset):
        def __init__(self, df, trans, **args):
            self.df = df
            self.trans = trans
            for k, v in args.items():
                setattr(self, k, v)
        
        def __getitem__(self, idx):
            image = np.array(Image.open(self.df.loc[idx, "file_name"]).convert(mode = "RGB"))
            label = self.df.loc[idx, "label"]
            if not isinstance(label, list):
                label_ = np.ones(self.num_classes) * self.smoothing / (self.num_classes - 1)
                label_[label] = 1 - self.smoothing
                label = label_
            label = np.array(label)
            if self.trans is not None:
                image = self.trans(image = image)["image"]
            return image, label

        def __len__(self):
            return len(self.df)

    def prepare_data(self):
        file_names = sorted(glob.glob("./data/train/*/*.jpg"))
        df = pd.DataFrame({"file_name": file_names})
        df["label"] = df.file_name.apply(lambda x: int(os.path.basename(os.path.dirname(x))))
        split = StratifiedKFold(5, shuffle = True, random_state = 0)
        train_idx, valid_idx = list(split.split(df, y = df.label))[self.fold]
        self.df_train = df.loc[train_idx].reset_index(drop = True) if self.fold != -1 else df.reset_index(drop = True)
        self.df_valid = df.loc[valid_idx].reset_index(drop = True)
        if self.classes is not None:
            self.df_train = self.df_train[self.df_train.label.isin(range(self.classes))].reset_index(drop = True)
            self.df_valid = self.df_valid[self.df_valid.label.isin(range(self.classes))].reset_index(drop = True)
        if self.pseudo_from:
            file_names = sorted(glob.glob("./data/test_B/*.jpg"))
            label = np.load(self.pseudo_from)
            df = pd.DataFrame({"file_name": file_names, "label": label.tolist()})
            self.df_train = pd.concat([self.df_train, df]).reset_index(drop = True)
        self.ds_train = self.Data(self.df_train, self.trans_train, **self.args)
        self.ds_valid = self.Data(self.df_valid, self.trans_valid, **self.args)

    def train_dataloader(self):
        return DataLoader(self.ds_train, self.batch_size, shuffle = True, num_workers = 4)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, self.batch_size, num_workers = 4)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.learning_rate, weight_decay = 2e-5)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.learning_rate, steps_per_epoch = int(len(self.train_dataloader())), epochs = self.num_epochs, anneal_strategy = "linear", final_div_factor = 30,), 'name': 'learning_rate', 'interval':'step', 'frequency': 1}
        return [optimizer], [lr_scheduler]

    def on_fit_start(self):
        metric_placeholder = {"valid_metric": 0}
        self.logger.log_hyperparams(self.hparams, metrics = metric_placeholder)

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
        self.log("valid_loss", loss, prog_bar = True)
        return y, yhat

    def validation_step_end(self, output):
        return output

    def validation_epoch_end(self, outputs):
        y = torch.cat([_[0] for _ in outputs]).argmax(1).detach().cpu().numpy()
        yhat = torch.cat([_[1] for _ in outputs]).argmax(1).detach().cpu().numpy()
        acc = accuracy_score(y, yhat)
        self.log("valid_metric", acc, prog_bar = True)

args = dict(
    learning_rate = 1e-3,
    model_name = "eca_nfnet_l1",
    num_epochs = 30,
    batch_size = 64,
    fold = -1,
    num_classes = 137,
    smoothing = 0.0,
    classes = None,
    alpha = 0.1,
    image_size = 384,
    swa = False,
    pseudo_from = "./data/pseudo/sub13.npy",
    drop_rate = 0.1,
    name = "pseudo/nfl1_fitter",
    version = "sorted_all"
)
args['trans_train'] = A.Compose([
    A.Resize(args['image_size'], args['image_size']),
    # A.HorizontalFlip(),
    # A.OneOf([
    #     A.RandomBrightnessContrast(),
    #     A.HueSaturationValue(),
    # ], p = 0.9),
    # A.OneOf([
    #     A.GridDistortion(),
    #     A.OpticalDistortion(),
    # ], p = 0.9),
    # A.Normalize(),
    # ToTensorV2()])
    A.load("./autoalbu/configs/outputs/2021-08-20/17-51-27/policy/latest.json")])
args['trans_valid'] = A.Compose([
    A.Resize(args['image_size'], args['image_size']),
    A.Normalize(),
    ToTensorV2()])

if __name__ == "__main__":
    logger = TensorBoardLogger("./logs", name = args["name"], version = args["version"], default_hp_metric = False)
    callback = pl.callbacks.ModelCheckpoint(
        filename = '{epoch}_{valid_metric:.3f}',
        save_last = True,
        mode = "max",
        monitor = 'valid_metric'
    )
    model = Model(**args)
    trainer = pl.Trainer(
        gpus = len(gpus.split(",")), 
        precision = 16, amp_backend = "native", amp_level = "O1", 
        accelerator = "dp",
        gradient_clip_val = 10,
        max_epochs = args["num_epochs"],
        stochastic_weight_avg = args["swa"],
        logger = logger,
        progress_bar_refresh_rate = 10,
        callbacks = [callback]
    )
    trainer.fit(model)
