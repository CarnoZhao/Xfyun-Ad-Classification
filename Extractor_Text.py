from Solver_Text import *
ab = "B"
val = ""
args["fold"] = 0

class ModelPred(Model):
    def __init__(self, **args):
        super(ModelPred, self).__init__(**args)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        return yhat

    def predict_dataloader(self):
        file_names = sorted(glob.glob(f"./data/test_{ab}/*.jpg"))
        df_test = pd.DataFrame({"file_name": file_names})
        df_test["label"] = 0
        df_test = df_test.merge(pd.read_csv(f"./data/test{ab}.tsv", sep = "\t"), on = "file_name")
        df_test = df_test.fillna("")
        if val:
            df_test = self.df_valid
        self.ds_test = self.Data(df_test, self.tokenizer, is_train = False, **self.args)
        return DataLoader(self.ds_test, self.batch_size, num_workers = 4)

trainer = pl.Trainer(
    gpus = len(gpus.split(",")), 
    precision = 16, amp_backend = "native", amp_level = "O1", 
    accelerator = "dp",
    logger = None,
    progress_bar_refresh_rate = 1
)

ckpts = [
    "./logs/text/rbt/fold0/checkpoints/epoch=25_valid_metric=0.810.ckpt",
    "./logs/text/rbt/fold1/checkpoints/epoch=28_valid_metric=0.806.ckpt",
    "./logs/text/rbt/sorted_all/checkpoints/epoch=28_valid_metric=0.956.ckpt"
]

preds = []
for ckpt in ckpts:
    model = ModelPred(**args)
    model = model.load_from_checkpoint(ckpt)
    pred = trainer.predict(model)
    pred = torch.cat(pred).softmax(1).detach().cpu().numpy()
    preds.append(pred)

preds = np.stack(preds)
np.save(f"./data/features/{'valid' if val else ('test' + ab)}_tex.npy", preds)