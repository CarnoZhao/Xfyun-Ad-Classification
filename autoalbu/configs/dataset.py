import torch.utils.data
import os
import glob
import pandas as pd
import numpy as np
import cv2
from PIL import Image


class SearchDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        # Implement additional initialization logic if needed
        file_names = glob.glob(os.path.join(os.path.basename(os.getcwd()), "data/train/*/*.jpg"))
        df = pd.DataFrame({"file_name": file_names})
        df["label"] = df.file_name.apply(lambda x: int(os.path.basename(os.path.dirname(x))))
        df = df[df.label < 60].reset_index(drop = True)
        idx = np.random.choice(len(df), 6000, replace = False)
        df = df.iloc[idx].reset_index(drop = True)
        self.df = df

    def __len__(self):
        # Replace `...` with the actual implementation
        return len(self.df)

    def __getitem__(self, index):
        # Implement logic to get an image and its label using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `label` should be an integer in the range [0, model.num_classes - 1] where `model.num_classes`
        # is a value set in the `search.yaml` file.

        image = np.array(Image.open(self.df.loc[index, "file_name"]).convert(mode = "RGB").resize((300, 300)))
        label = np.array(self.df.loc[index, "label"])

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
