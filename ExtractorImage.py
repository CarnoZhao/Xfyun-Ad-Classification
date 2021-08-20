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
        # ds_test = self.ds_valid
        return DataLoader(ds_test, self.batch_size, num_workers = 4)

trainer = pl.Trainer(
    gpus = len(gpus.split(",")), 
    precision = 16, amp_backend = "native", amp_level = "O1", 
    accelerator = "dp",
    logger = None,
    progress_bar_refresh_rate = 1
)

ckpts = [
    # "wandb/run-20210701_145413-effv2s_drop0.3_mix0.4_imsz384_0/files/baseline/effv2s_drop0.3_mix0.4_imsz384_0/checkpoints/epoch=22_valid_metric=0.860.ckpt",
    # "./wandb/offline-run-20210715_191940-effv2s_drop0.3_mix0.4_imsz512_augv3_noclip_0/files/baseline/effv2s_drop0.3_mix0.4_imsz512_augv3_noclip_0/checkpoints/epoch=30_valid_metric=0.866.ckpt",
    "./wandb/offline-run-20210718_214855-effv2s_drop0.3_mix0.4_imsz384_augv3_noclip_0/files/baseline/effv2s_drop0.3_mix0.4_imsz384_augv3_noclip_0/checkpoints/epoch=30_valid_metric=0.870.ckpt",
    "./wandb/offline-run-20210719_095922-b3ns_drop0.3_mix0.4_imsz384_augv3_noclip_0/files/baseline/b3ns_drop0.3_mix0.4_imsz384_augv3_noclip_0/checkpoints/epoch=30_valid_metric=0.869.ckpt",
    "./wandb/offline-run-20210706_124828-effv2s_drop0.3_mix0.4_imsz384_1/files/baseline/effv2s_drop0.3_mix0.4_imsz384_1/checkpoints/epoch=30_valid_metric=0.868.ckpt",
    # "./wandb/offline-run-20210713_085719-effv2s_drop0.3_mix0.4_imsz384_2/files/baseline/effv2s_drop0.3_mix0.4_imsz384_2/checkpoints/epoch=29_valid_metric=0.862.ckpt"
]
# effv2s_drop0.3_mix0.4_imsz384_0 [0.6       0.88347148]
# b3_ns_drop0.3_mix0.2_imsz384_0 [0.61       0.88200878]
# b3_ns_drop0.3_imsz384_0        [0.52       0.88532423]
# b4_ns_drop0.3_mix0.4_imsz512_50ep_0 [0.54       0.88337396]

preds = []
for ckpt in ckpts:
    if "b4_ns" in ckpt:
        model_name = "tf_efficientnet_b4_ns"
    elif "b3_ns" in ckpt:
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
# merge_preds1 = np.concatenate([preds[:4] * alpha, tex_preds[:2] * (1 - alpha)], 0).sum(0)
merge_preds = preds[:4].mean(0) * alpha + tex_preds[:2].mean(0) * (1 - alpha)
sub = pd.DataFrame({"image_id": model.predict_dataloader().dataset.df.file_name, "category_id": merge_preds.argmax(1)})
sub.image_id = sub.image_id.apply(lambda x: x.split("/")[-1])
sub.to_csv("submission.csv", index = False)

tmp = preds[1] * 0.6 + tex_preds[0] * 0.4
accuracy_score(model.predict_dataloader().dataset.df.label, tmp.argmax(1))

cm1 = confusion_matrix(model.predict_dataloader().dataset.df.label, (preds[0] * 0.6 + tex_preds[0] * 0.4).argmax(1))
ac1 = cm1.sum(1) - (cm1).diagonal()
ac1
# np.unravel_index(np.argmax(cm1 - cm1*np.eye(len(cm1))), cm1.shape)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize = (6, 6))

# ax.imshow(cm1 - cm1 * np.eye(len(cm1)))

# fig.savefig("tmp.png")
# plt.close(fig)

# cm2 = confusion_matrix(model.predict_dataloader().dataset.df.label, tex_preds[0].argmax(1))
# ac2 = (cm2).diagonal()
# alpha = 1 / (1 + np.exp(ac1 - ac2))
# (preds * alpha + tex_preds * (1 - alpha)).sum(0)

# gs = []
# for alpha in np.arange(0.5, 0.7, 0.01):
#     merge_preds = (preds * alpha + tex_preds * (1 - alpha)).sum(0)
#     sub = pd.DataFrame({"image_id": model.predict_dataloader().dataset.df.file_name, "category_id": merge_preds.argmax(1)})
#     acc = accuracy_score(sub.category_id, model.predict_dataloader().dataset.df.label)
#     gs.append([alpha, acc])
# gs = np.array(gs)
# alpha = gs[:,0][gs[:,1].argmax()]
# print(gs[gs[:,1].argmax()])


# alpha = 0.6
# merge_preds = (preds * alpha + tex_preds * (1 - alpha)).sum(0)
# sub = pd.DataFrame({"image_id": model.predict_dataloader().dataset.df.file_name, "category_id": merge_preds.argmax(1)})
