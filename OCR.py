from chineseocr_lite.model import text_predict
import glob
import cv2
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image

texts = []
for file_name in tqdm(sorted(glob.glob("./data/train/*/*.*"))):
    img = np.array(Image.open(file_name).convert(mode = "RGB"))
    pred = text_predict(img)
    text = "，".join([_["text"] for _ in pred])
    texts.append([file_name, text])

res = pd.DataFrame(texts, columns = ["file_name", "text"])
res.to_csv("./data/train.tsv", index = False, sep = "\t")

texts = []
for file_name in tqdm(sorted(glob.glob("./data/test_B/*.*"))):
    img = np.array(Image.open(file_name).convert(mode = "RGB"))
    pred = text_predict(img)
    text = "，".join([_["text"] for _ in pred])
    texts.append([file_name, text])

res = pd.DataFrame(texts, columns = ["file_name", "text"])
res.to_csv("./data/test_B.tsv", index = False, sep = "\t")
