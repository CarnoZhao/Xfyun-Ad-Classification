# Xfyun-Ad-Classification

Codes for [**iFLYTEK AI开发者大赛 广告图片素材分类算法挑战赛**](https://challenge.xfyun.cn/topic/info?type=ad-2021)

# Environment

```sh
python=3.8.0
torch=1.7.1
pytorch_lightning=1.3.5
timm=0.4.9
cuda=11.0
```

# External Utils

- [chineseocr_lite](https://github.com/DayBreak-u/chineseocr_lite/tree/master) (*modified by myself*)

- [my own deep learning utils](https://github.com/CarnoZhao/utils)

# Usage

1. download datasets from official competition site

2. run `OCR.py` to get unordered texts from images (*run twice for training data and test data*)

3. `SolverImage.py` for image classification training

4. `SolverText.py` for text classification training

5. `ExtractorText.py` for text softmax probabilities extraction

6. `ExtractorImage.py` for image softmax probabilities extraction and Image-Text multi-modality merge (simply weighted average)

# Reproduce

## 0. Note:

### 0.1 Reproducibility

Although I have fixed random seeds as many as possible in main training codes, I did not test the reproducibility of external open source codes in `OCR` and `autoalbu`. This pipeline was expected/estimated to get score *0.907~0.909* (rank3) on semi-final leaderboard.

### 0.2 Test data

All codes were writen with *semi-final test dataset*, named `test_B` or `testB` by myself.

### 0.3 GPU Deivces

- For **training**, these codes were run under *2x TITAN RTX* or *2x RTX 3090*. At least *2x 24G = 48G* GPU memory is required for best reproduction.

- For **inferencing only**, *6-8G* GPU is enough.


## 1. Data preprocessing

### 1.1 Download data to `./data`, arranged as:

```
|--data
   |--train
   |   |--0
   |   |  |--xxx.jpg
   |   |  |--xxx.jpg 
   |   |
   |   |--1
   |   |--...
   |
   |--test_B
       |--xxx.jpg
       |--xxx.jpg
```

### 1.2 OCR on image data

- OCR codes were modified from open source github repo (`master` branch): [chineseocr_lite](https://github.com/DayBreak-u/chineseocr_lite.git)  (Thanks to authors)

- OCR base codes are located at `./chineseocr_lite`. Compared with original repo, I only set configs in `./chineseocr_lite/config.py`, and removed useless codes and models. The changes are not included in this repo's history.

- (**RUN**) 

```
python OCR.py
```

This is to extract text information from `train` and `test_B` datasets. The results will be saved in `./data/train.tsv` and `./data/testB.tsv`.

> NOTE: I am not an expert in OCR, so I did not pay much attention in OCR model selection, but chose the easiest one from github.

### 1.3 Auto-albumentations 

- For detailed usage and installation, please refer to [autoalbument](https://albumentations.ai/docs/autoalbument/)

    - There might be some conflits between `autoalbu`'s requirement of `pytorch-lightning` and the `pytorch-lightning` version of this repo. Please create an independent environment for `autoalbu` according to its own installation instruction.

- The auto-albu codes are located at `./autoalbu`

- Set the necessary `dataset.py` and `search.yaml` in `./autoalbu/configs`

- (**RUN**) 
```
cd ./autoalbu

autoalbument-search --config-dir configs
```

- Because the loss did not decrease in `autoalbu` training after 15 epochs, to save time, I manually stopped `autoalbu` training at 17 epoch. The results will be saved in `./autoalbu/configs/outputs/{your_run_date}/{your_run_time}/policy/`

- For better git repo performance, I set `.gitignore` to ignore other outputs and moved the `./autoalbu/configs/outputs/{your_run_date}/{your_run_time}/policy/lasted.json` to `./autoalbu/configs/latest.json`.

## 2. Model training

### 2.1 Main Idea

- Train image models and text models respectively

- Ensemble image models and text models, and test ensemble results on leaderboard

- Following the most basic **knowledge distilation** and **semi-supervised learning** strategy:

    - use the softmax probability of ensemble prediction and training ground truth to train a smaller image model and a smaller text model.

    - **knowledge distilation**: use larger model ensemble to train smaller model

    - **semi-supervised learning**: use soft-pseudo-label of unlabeled data to train model

- Ensemble the distilled image model and text model as final output

### 2.2 Image model training

- Models (from [timm](https://github.com/rwightman/pytorch-image-models.git)):

    - `tf_efficientnet_b4_ns`

    - `tf_efficientnetv2_m_in21ft1k`

    - `eca_nfnet_l1`

    - `resnet200d`

    - `swin_base_patch4_window12_384`

- Hyperparameters

    ```python
    args = dict(
        seed = 0,
        learning_rate = 1e-3,
        model_name = #?#,
        num_epochs = 30,
        batch_size = 64,
        fold = -1,         # use all training data
        num_classes = 137, # number of classes
        smoothing = 0.1,   # label smoothing
        classes = None,    # useless
        alpha = 0.4,       # mixup alpha
        swa = True,         # use stochastic weight average?
        image_size = #?#,  # training image size
        drop_rate = 0.3,   # dropout rate
        name = "image/#?#",    # logs saving directory
        version = "sorted_all" # logs saving directory
    )
    ```

    - `args['model_name']` are five image model names mentioned above.

    - `args['image_size']` are **512** for `tf_efficientnet_b4_ns` and  `tf_efficientnetv2_m_in21ft1k`, and **384** for other three models.

    - `args['name']` should be different for five models. Here are my names:    

        - `tf_efficientnet_b4_ns`: "image/b4ns"

        - `tf_efficientnetv2_m_in21ft1k`: "image/ev2m"

        - `eca_nfnet_l1`: "image/nfl1"

        - `resnet200d`: "image/200d"

        - `swin_base_patch4_window12_384`: "image/swb"

- (**RUN**)

```
# set hyperparameters inside Solver_Image.py

python Solver_Image.py # should be run for five times

# all logs will be saved at "./logs/image/{args['name']}"
```

- Saved checkpoints for future usage:

    
    - `./logs/image/nfl1/sorted_all/checkpoints/epoch=22_valid_metric=0.983.ckpt`

    - `./logs/image/b4ns/sorted_all/checkpoints/epoch=28_valid_metric=0.990.ckpt`

    - `./logs/image/ev2m/sorted_512_all/checkpoints/epoch=22_valid_metric=0.987.ckpt`

    - `./logs/image/200d/sorted_all/checkpoints/epoch=29_valid_metric=0.987.ckpt`

    - `./logs/image/swb/sorted_all/checkpoints/epoch=29_valid_metric=0.971.ckpt`


### 2.3 Text model training

- Models (from [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)):

    - `hfl/chinese-roberta-wwm-ext`

    - `hfl/chinese-bert-wwm-ext`

    - `hfl/chinese-bert-wwm`

- Hyperparameters

    ```python
    args = dict(
        learning_rate = 2e-5,
        model_name = #?#,
        num_epochs = 30,
        batch_size = 64,
        fold = -1,
        num_classes = 137,
        smoothing = 0.1,
        alpha = 0,
        max_length = 256,
        drop_rate = 0.3,
        swa = #?#,
        name = "text/#?#",
        version = "sorted_all"
    )
    ```

    - `args['model_name']` are three text model names mentioned above.

    - `args['swa']` was set to `False` in `hfl/chinese-bert-wwm` (due to some bugs of my machine)

    - `args['name']` should be different for three models. Here are my names:    

        - `hfl/chinese-roberta-wwm-ext`: "rbt"

        - `hfl/chinese-bert-wwm-ext`: "bt"

        - `hfl/chinese-bert-wwm`: "btwwm"

- (**RUN**)

```
# set hyperparameters inside Solver_Text.py

python Solver_Text.py # should be run for three times


# all logs will be saved at "./logs/text/{args['name']}"
```

- Saved checkpoints for future usage:

    - `./logs/text/rbt/sorted_all/checkpoints/epoch=28_valid_metric=0.956.ckpt`

    - `./logs/text/bt/sorted_all/checkpoints/epoch=29_valid_metric=0.966.ckpt`

    - `./logs/text/btwwm/sorted_all/checkpoints/epoch=27_valid_metric=0.965.ckpt`

## 3. Multi-modality model ensemble

### 3.1 Extract softmax probability from models

- Set model checkpoint paths inside `Extractor_Image.py`

- Set model checkpoint paths inside `Extractor_Text.py`

- (**RUN**)

```
mkdir -p ./data/features
python Extractor_Image.py
python Extractor_Text.py
```

- The extracted features will be saved at `./data/features`

### 3.2 Ensemble features

- Set saved features paths inside `Ensembler.py`

- The ensemble procedure only used the simplest weighted average, because other complex method will decrease the accuracy performance.

- (**RUN**)

```
mkdir -p ./data/pseudo
python Ensembler.py
```

- The ensembled soft-pseudo-label will be saved at `./data/pseudo`

## 4.  Knowledge distilation + Semi-supervised learning

- Set the peuso-label numpy array path inside `Solver_Pseudo.py`

- To balanced the inaccurate prediction of pseudo-label, we used less mixup, dropout and label smoothing.

- (**RUN**)

```
python Solver_Pseudo.py
```
