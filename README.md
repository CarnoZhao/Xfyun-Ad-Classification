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

# How to run

1. download datasets from official competition site

2. run `OCR.py` to get unordered texts from images (*run twice for training data and test data*)

3. `SolverImage.py` for image classification training

4. `SolverText.py` for text classification training

5. `ExtractorText.py` for text softmax probabilities extraction

6. `ExtractorImage.py` for image softmax probabilities extraction and Image-Text multi-modality merge (simply weighted average)


