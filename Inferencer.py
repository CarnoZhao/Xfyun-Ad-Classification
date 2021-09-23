import Extractor_Image as ei
import Extractor_Text as et
torch = ei.torch
np = ei.np
pd = ei.pd
os = ei.os

if __name__ == "__main__":
    model_image = ei.ModelPred(tta = False, **ei.args)
    model_text = et.ModelPred(**et.args)

    ckpt_image = "./weights/pseudo_b0ns.ckpt"
    ckpt_text = "./weights/pseudo_rbt.ckpt"

    model_image = model_image.load_from_checkpoint(ckpt_image)
    model_text = model_text.load_from_checkpoint(ckpt_text)

    pred_image = ei.trainer.predict(model_image)
    pred_image = torch.cat(pred_image).softmax(1).detach().cpu().numpy()
    pred_text = et.trainer.predict(model_text)
    pred_text = torch.cat(pred_text).softmax(1).detach().cpu().numpy()

    pred = (0.6 * pred_image + 0.4 * pred_text).argmax(1)
    image_id = model_image.predict_dataloader().dataset.df.file_name.apply(os.path.basename)
    sub = pd.DataFrame({"image_id": image_id, "category_id": pred})
    sub.to_csv("submission.csv", index = False)