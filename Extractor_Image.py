from Solver_Image import *
from sklearn.metrics import confusion_matrix, f1_score
ab = "B"
val = True
args["fold"] = 0

class ModelPred(Model):
    classes = None
    def __init__(self, **args):
        super(ModelPred, self).__init__(**args)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        yhat = 0
        xs = [x, x.flip(-1)]
        for x in xs:
            yhat += self(x) / len(xs)
        return yhat

    def predict_dataloader(self):
        file_names = sorted(glob.glob(f"./data/test_{ab}/*.jpg"))
        df_test = pd.DataFrame({"file_name": file_names})
        df_test["label"] = 0
        if val:
            df_test = self.df_valid
        self.ds_test = self.Data(df_test, self.trans_valid, is_train = False, **self.args)
        return DataLoader(self.ds_test, self.batch_size, num_workers = 4)

trainer = pl.Trainer(
    gpus = len(gpus.split(",")), 
    precision = 16, amp_backend = "native", amp_level = "O1", 
    accelerator = "dp",
    logger = None,
    progress_bar_refresh_rate = 1
)

ckpts = [
    # "./logs/image/effv2m/fold0/checkpoints/epoch=22_valid_metric=0.869.ckpt",
    # "./logs/image/effv2m/fold1/checkpoints/epoch=30_valid_metric=0.870.ckpt",
    # "./logs/image/ev2s/fold0/checkpoints/epoch=28_valid_metric=0.864.ckpt",
    # "./logs/image/ev2s/fold1/checkpoints/epoch=30_valid_metric=0.864.ckpt",
    # "./logs/image/nfl1/sorted_all/checkpoints/epoch=22_valid_metric=0.983.ckpt",
    # "./logs/image/ev2s/sorted_all/checkpoints/epoch=28_valid_metric=0.989.ckpt"
    "./logs/image/nfl1/sorted_fold0/checkpoints/epoch=29_valid_metric=0.872.ckpt"
]

preds = []
for ckpt in ckpts:
    model = ModelPred(**args)
    model = model.load_from_checkpoint(ckpt, strict = False)
    pred = trainer.predict(model)
    pred = torch.cat(pred).softmax(1).detach().cpu().numpy()
    preds.append(pred)

preds = np.stack(preds)
np.save(f"./data/features/{'valid' if val else ('test' + ab)}_img.npy", preds)

preds_img = np.load(f"./data/features/{'valid' if val else ('test' + ab)}_img.npy")
preds_tex = np.load(f"./data/features/{'valid' if val else ('test' + ab)}_tex.npy")

preds = preds_img[-1:].mean(0) * 0.6# + preds_tex.mean(0) * 0.4
preds = preds.argmax(1)
gt = model.predict_dataloader().dataset.df.label
accuracy_score(gt, preds)

sub = pd.DataFrame({"image_id": model.predict_dataloader().dataset.df.file_name, "category_id": preds})
sub.image_id = sub.image_id.apply(lambda x: x.split("/")[-1])
sub.to_csv("submission.csv", index = False)
