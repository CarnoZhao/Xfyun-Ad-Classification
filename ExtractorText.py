gpus = "0"
learning_rate = 1e-4
batch_size = 64
n_epochs = 30
model_name = "hfl/chinese-roberta-wwm-ext"
image_size = 384
fold = 0
drop_rate = 0.3
num_classes = 137
smooth = 0.1
alpha = 0
max_length = 256
long_resize = False
imbalance_sample = False

import re
import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
from SolverText import Model
from sklearn.metrics import confusion_matrix

class ModelExtract(Model):
    def __init__(self, **kwargs):
        super(ModelExtract, self).__init__(**kwargs)
        # self.model.classifier = nn.Identity()

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        yhat = self(x)
        return yhat

    def predict_dataloader(self):
        image_files = sorted(glob.glob("./data/test/*.*"))
        df = pd.DataFrame(image_files, columns = ["file_name"])
        df["label"] = 0 # df.file_name.apply(lambda x: int(x.split('/')[-2]))
        df = df.merge(pd.read_csv("./data/test.tsv", sep = "\t"), on = "file_name")
        df = df.fillna("")
        df.text = df.text.apply(lambda x: re.sub(r"[^\u4e00-\u9fef]", "", x))
        ds_test = self.Data(df, self.tokenizer, self.max_length)
        # ds_test = self.ds_valid
        return DataLoader(ds_test, self.batch_size, num_workers = 4)

model = ModelExtract(
    learning_rate = learning_rate, 
    batch_size = batch_size, 
    n_epochs = n_epochs, 
    model_name = model_name, 
    image_size = image_size, 
    fold = fold, 
    drop_rate = drop_rate, 
    num_classes = num_classes, 
    smooth = smooth, 
    train_trans = None, 
    valid_trans = None,
    alpha = alpha,
    imbalance_sample = imbalance_sample,
    criterion = None,
    long_resize = long_resize,
    max_length = max_length
)
trainer = pl.Trainer(
    gpus = len(gpus.split(",")), 
    precision = 16, amp_backend = "native", amp_level = "O1", 
    accelerator = "dp",
    logger = None,
    progress_bar_refresh_rate = 1
)

ckpts = [_ for _ in glob.glob("./wandb/**/*.ckpt", recursive = True) if re.search("(?=.*rbt_.*)(?=.*last.*)", _) is not None and re.search("(noclip|latest)", _) is None]
preds = []
for ckpt in ckpts:
    model = model.load_from_checkpoint(ckpt)
    pred = trainer.predict(model)
    pred = torch.cat(pred).softmax(1)
    pred = pred.detach().cpu().numpy()
    preds.append(pred)
preds = np.stack(preds)

np.save("./data/test_tex.npy", preds)


# df = model.predict_dataloader().dataset.df
# for i in range(len(pred[0])):
#     df[f"tex{i}"] = pred[:,i]
# df.to_pickle("./data/test_tex.pkl")

# cm = confusion_matrix(pred.argmax(1), df.label)
# cm.sum(0)
# np.unique(df.label, return_counts = True)
# (cm * (1 - np.eye(137, dtype = int))).sum(0)