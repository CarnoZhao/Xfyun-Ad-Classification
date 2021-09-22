from Solver_Image import *
from sklearn.metrics import confusion_matrix, f1_score
ab = "B"
val = False
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
    "./logs/image/nfl1/sorted_all/checkpoints/epoch=22_valid_metric=0.983.ckpt",
    "./logs/image/b4ns/sorted_all/checkpoints/epoch=28_valid_metric=0.990.ckpt",
    "./logs/image/ev2m/sorted_512_all/checkpoints/epoch=22_valid_metric=0.987.ckpt",
    "./logs/image/200d/sorted_all/checkpoints/epoch=29_valid_metric=0.987.ckpt",
    "./logs/image/swb/sorted_all/checkpoints/epoch=29_valid_metric=0.971.ckpt"
]

model = ModelPred(**args)
model.prepare_data()

preds = []
for ckpt in ckpts:
    model = ModelPred(**args)
    model = model.load_from_checkpoint(ckpt, strict = False)
    pred = trainer.predict(model)
    pred = torch.cat(pred).softmax(1).detach().cpu().numpy()
    np.save(f"./data/features/{'valid' if val else ('test' + ab)}_{ckpt.split('/')[2]}_{ckpt.split('/')[3]}_{ckpt.split('/')[4]}.npy", pred)
    preds.append(pred)


