gpus = "0"
learning_rate = 1e-3
batch_size = 64
n_epochs = 30
model_name = "tf_efficientnet_b3_ns"
image_size = 384
fold = 0
drop_rate = 0.3
num_classes = 137
smooth = 0.1
alpha = 0.4
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
from SolverImage import Model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class ModelExtract(Model):
    def __init__(self, **kwargs):
        super(ModelExtract, self).__init__(**kwargs)
        # self.model.classifier = nn.Identity()

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        yhat = 0
        xs = [x, x.flip(-1)]
        for x in xs:
            yhat += self(x) / len(xs)
        return yhat

    def predict_dataloader(self):
        # file_names = glob.glob("./data/train/*/*.jpg")
        file_names = sorted(glob.glob("./data/test/*.jpg"))
        df_test = pd.DataFrame({"file_name": file_names})
        df_test["label"] = 0
        test_trans = A.Compose([
            A.Resize(image_size, image_size) if not long_resize else A.LongestMaxSize(image_size),
            A.PadIfNeeded(image_size, image_size),
            A.Normalize()])
        ds_test = self.Data(df_test, test_trans)
        ds_test = self.ds_valid
        return DataLoader(ds_test, self.batch_size, num_workers = 4)

trainer = pl.Trainer(
    gpus = len(gpus.split(",")), 
    precision = 16, amp_backend = "native", amp_level = "O1", 
    accelerator = "dp",
    logger = None,
    progress_bar_refresh_rate = 1
)

ckpts = [_ for _ in glob.glob("./wandb/**/*.ckpt", recursive = True) if re.search("(?=(effv2s|b3ns)_drop0.3_mix0.4_imsz384_augv3_noclip)(?!.*ep_.*)(?!.*last.*)", _) is not None]

preds = []
for ckpt in ckpts:
    if "b3_ns" in ckpt:
        model_name = "tf_efficientnet_b3_ns"
    else:
        model_name = "tf_efficientnetv2_s_in21ft1k"
    if "imsz384" in ckpt:
        image_size = 384
    else:
        image_size = 512
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
        mask = "mask" in model_name
    )
    model = model.load_from_checkpoint(ckpt)
    model.classes = None
    pred = trainer.predict(model)
    pred = torch.cat(pred)
    pred = pred.softmax(1).detach().cpu().numpy()
    preds.append(pred)
preds = np.stack(preds)

tex_preds = np.load("./data/test_tex.npy")

alpha = 0.6
merge_preds = preds.mean(0) * alpha# + tex_preds.mean(0) * (1 - alpha)
sub = pd.DataFrame({"image_id": model.predict_dataloader().dataset.df.file_name, "category_id": merge_preds.argmax(1)})
sub.image_id = sub.image_id.apply(lambda x: x.split("/")[-1])
sub.to_csv("submission.csv", index = False)

cm1 = confusion_matrix(model.predict_dataloader().dataset.df.label, (preds[0] * 0.6 + tex_preds[0] * 0.4).argmax(1))
ac1 = cm1.sum(1) - (cm1).diagonal()


from torchcam.cams import SmoothGradCAMpp
cam_extractor = SmoothGradCAMpp(model, "model.act2")
model.eval()

for x, y in model.predict_dataloader():
    break