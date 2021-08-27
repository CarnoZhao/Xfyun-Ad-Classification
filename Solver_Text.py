gpus = "0,1"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
import warnings
warnings.filterwarnings("ignore")

import re
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils.loss.smooth import LabelSmoothingLoss
from utils.mixup import mixup_data, mixup_criterion
pl.seed_everything(0)

class Model(pl.LightningModule):
    def __init__(self, **args):
        super(Model, self).__init__()
        for k, v in args.items():
            setattr(self, k, v)
        self.args = args
        self.model = transformers.BertForSequenceClassification.from_pretrained(self.model_name)
        self.model.dropout = nn.Dropout(self.drop_rate)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, self.num_classes)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.model_name)
        self.criterion = LabelSmoothingLoss(classes = self.num_classes, smoothing = self.smoothing)
        self.save_hyperparameters()

    class Data(Dataset):
        def __init__(self, df, trans, is_train, **args):
            self.df = df
            self.trans = trans
            self.is_train = is_train
            for k, v in args.items():
                setattr(self, k, v)
        
        def __getitem__(self, idx):
            label = self.df.loc[idx, "label"]
            text = self.df.loc[idx, "text"]
            if self.is_train:
                text = text.split("，")
                np.random.shuffle(text)
                text = "".join(text)
            text = re.sub(r"[^\u4e00-\u9fef]", "", text)
            tok = self.trans.encode_plus(
                text,
                add_special_tokens = True,
                truncation = 'longest_first',
                max_length = self.max_length,
                padding="max_length")
            tok = {k: np.array(v).astype(np.long) for k, v in tok.items()}
            return tok, label

        def __len__(self):
            return len(self.df)

    def prepare_data(self):
        file_names = sorted(glob.glob("./data/train/*/*.jpg"))
        df = pd.DataFrame({"file_name": file_names})
        df["label"] = df.file_name.apply(lambda x: int(os.path.basename(os.path.dirname(x))))
        split = StratifiedKFold(5, shuffle = True, random_state = 0)
        train_idx, valid_idx = list(split.split(df, y = df.label))[self.fold]
        df = df.merge(pd.read_csv("./data/train.tsv", sep = "\t"), on = "file_name")
        df = df.fillna("")
        # df.text = df.text.apply(lambda x: re.sub(r"[^\u4e00-\u9fef]", "", x))
        self.df_train = df.loc[train_idx].reset_index(drop = True) if self.fold != -1 else df.reset_index(drop = True)
        self.df_valid = df.loc[valid_idx].reset_index(drop = True)
        self.ds_train = self.Data(self.df_train, self.tokenizer, is_train = True, **self.args)
        self.ds_valid = self.Data(self.df_valid, self.tokenizer, is_train = False, **self.args)

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
        out = self.model(**x)["logits"]
        return out

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
        y = torch.cat([_[0] for _ in outputs]).detach().cpu().numpy()
        yhat = torch.cat([_[1] for _ in outputs]).argmax(1).detach().cpu().numpy()
        acc = accuracy_score(y, yhat)
        self.log("valid_metric", acc, prog_bar = True)

args = dict(
    learning_rate = 2e-5,
    model_name = "hfl/chinese-roberta-wwm-ext",
    num_epochs = 30,
    batch_size = 64,
    fold = -1,
    num_classes = 137,
    smoothing = 0.1,
    alpha = 0,
    max_length = 256,
    drop_rate = 0.3,
    name = "text/rbt",
    version = "sorted_all"
)

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
        stochastic_weight_avg = True,
        logger = logger,
        progress_bar_refresh_rate = 10,
        callbacks = [callback]
    )
    trainer.fit(model)
