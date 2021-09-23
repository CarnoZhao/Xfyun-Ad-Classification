# Xfyun-Ad-Classification

Codes for [**iFLYTEK AI开发者大赛 广告图片素材分类算法挑战赛**](https://challenge.xfyun.cn/topic/info?type=ad-2021)

# Environment

- install by conda or pip

```sh
cuda=10.2
python=3.8.0 # installed by conda
torch=1.7.1  # installed by conda
pytorch_lightning=1.4.5 # intalled by pip
timm=0.4.9              # intalled by pip 
transformers=4.9.2      # installed by pip
albumentations=1.0.0    # installed by pip
```

# External Utils

- [chineseocr_lite](https://github.com/DayBreak-u/chineseocr_lite/tree/master) (*modified by myself*)

- [my own deep learning utils](https://github.com/CarnoZhao/utils)

# Usage

1. Datasets from official competition site

2. `OCR.py` for unordered texts from images (*run twice for training data and test data*)

3. `Solver{Image|Text}.py` for image/text classification training

4. `Extractor{Image|Text}.py` for image/text softmax probabilities extraction

5. `Ensembler.py` for multi-modality ensemble

6. `Solver_{Image|Text}_Pseudo.py` for pseudo-label training of image/text model

7. `Inferencer.py` for final prediction

# Reproduce

## 0. Note:

### 0.1 Reproducibility

Although I have fixed random seeds as many as possible in main training codes, I did not test the reproducibility of external open source codes in `OCR` and `autoalbu`. This pipeline was expected/estimated to get score *0.907~0.909* (rank3) on semi-final leaderboard.

### 0.2 Test data

All codes were writen with *semi-final test dataset*, named `test_B` or `testB` by myself.

### 0.3 GPU Deivces

- For **training**, these codes were run under *2x TITAN RTX* or *2x RTX 3090*. At least *2x 24G = 48G* GPU memory is required for best reproduction.

- For **inferencing only**, *6-8G* GPU is enough.

### 0.4 Training time

- Using *2x TITAN RTX*, the training time per image model was 4~6h, and the training time per text model was 1~2h.


## 1. Data preprocessing

### 1.1 Directory structure

- Download data to `./data`, arranged as:

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

- OCR base codes are located at `./chineseocr_lite`. Compared with original repo, I only changed configs in `./chineseocr_lite/config.py`, and removed useless codes and models. The changes are not included in this repo's history.

- (**RUN**) 

```
python OCR.py
```

This is to extract text information from `train` and `test_B` datasets. The results will be saved in `./data/train.tsv` and `./data/testB.tsv`.

> NOTE: I am not an expert in OCR, so I did not pay much attention in OCR model selection, but chose the easiest one from github.

### 1.3 Auto-albumentations 

- For detailed usage and installation, please refer to [autoalbument](https://albumentations.ai/docs/autoalbument/)

    - There might be some conflicts between `autoalbu`'s requirement of `pytorch-lightning` and the `pytorch-lightning` version of this repo. Please create an independent environment for `autoalbu` according to its own installation instruction.

- The auto-albu codes are located at `./autoalbu`

- Set the necessary `dataset.py` and `search.yaml` in `./autoalbu/configs`

- (**RUN**) 
```
cd ./autoalbu

autoalbument-search --config-dir configs
```

- Because the loss did not decrease in `autoalbu` training after 15 epochs, to save time, I manually stopped `autoalbu` training at 17 epoch. The results will be saved in `./autoalbu/configs/outputs/{your_run_date}/{your_run_time}/policy/`

- For better git repo performance, I set `.gitignore` to ignore other outputs and moved the `./autoalbu/configs/outputs/{your_run_date}/{your_run_time}/policy/lasted.json` to `./autoalbu/configs/latest.json`.

> NOTE:  When using learnt `autoalbu` policy, the path should be configured manually in training codes.

## 2. Model training

### 2.1 Main Idea

- Train image models and text models respectively

- Ensemble image models and text models, and test ensemble results on leaderboard

- Following the most basic **knowledge distillation** and **semi-supervised learning** strategy:

    - use the pseudo-label of ensemble prediction and training ground truth to train a smaller image model and a smaller text model.

    - **knowledge distillation**: use larger model ensemble to train smaller model

    - **semi-supervised learning**: use pseudo-label of unlabeled data to train model

- Ensemble the distilled image model and text model as final output

![pipeline](assets/pipeline.jpg)
_Pipeline_

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

- (**RUN FOR 5 TIMES**) 

    - Set hyperparameters inside Solver_Image.py 

    - Run command below for each model

```
python Solver_Image.py
```

- All logs will be saved at `./logs/image/{args['name']}`

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

    - `args['swa']` are set to `False` in `hfl/chinese-bert-wwm` (due to some bugs of my machine), and `True` in others.

    - `args['name']` should be different for three models. Here are my names:    

        - `hfl/chinese-roberta-wwm-ext`: "rbt"

        - `hfl/chinese-bert-wwm-ext`: "bt"

        - `hfl/chinese-bert-wwm`: "btwwm"

- (**RUN FOR 3 TIMES**) 

    - Set hyperparameters inside Solver_Text.py 

    - Run command below for each model

```
python Solver_Text.py
```

- All logs will be saved at `./logs/text/{args['name']}`


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

- The ensemble procedure only used the simplest weighted average, because other complex methods will decrease the accuracy performance.

- (**RUN**)

```
mkdir -p ./data/pseudo
python Ensembler.py
```

- The ensembled pseudo-label will be saved at `./data/pseudo`

## 4.  Knowledge distillation + Semi-supervised learning

- Set the peuso-label numpy array path inside `Solver_Pseudo.py`

- To balance the inaccurate prediction of pseudo-label, we used less augmentation, mixup, dropout and label smoothing.

- Model selection:

    - For image model, I chose a liter one: `tf_efficientnet_b0_ns`

    - For text model, since liter version `hfl/rbt3` did not perform well, I still used `hfl/chinese-roberta-wwm-ext`

- (**RUN**)

```
python Solver_Image_Pseudo.py
python Solver_Text_Pseudo.py
```

- The logs will be saved at `./logs/pseudo`

## 5. Final prediction

- Set pseudo-label trained model paths (image model & text model) inside `Inferencer.py`

- (**RUN**)

```
python Inferencer.py
```

- The prediction result will be saved at `./submission.csv`, formatted following the submission requirement.