from chineseocr_lite.model import text_predict
import glob
import cv2
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image

texts = []
for file_name in tqdm(sorted(glob.glob("./data/test_B/*.*"))):
    # os.system(f"mkdir -p {os.path.dirname(file_name).replace('/train/', '/masked_train/')}")
    img = np.array(Image.open(file_name).convert(mode = "RGB"))
    pred = text_predict(img)
    # for p in pred:
    #     img = cv2.fillPoly(img, [p["bbox"].reshape((-1, 1, 2)).astype(np.int32)], (0, 0, 0))
    # Image.fromarray(img).save(file_name.replace("/train/", "/masked_train/"))
    text = "，".join([_["text"] for _ in pred])
    texts.append([file_name, text])

res = pd.DataFrame(texts, columns = ["file_name", "text"])
res.to_csv("./data/test_B.tsv", index = False, sep = "\t")

# pred = text_predict(img)
# for p in pred:
#     tmp = cv2.fillPoly(img, [p["bbox"].reshape((-1, 1, 2)).astype(np.int32)], (0, 0, 0))
# Image.fromarray(tmp).save("tmp.png")