# Deep Learning SotA

This repository lists the state-of-the-art results for mainstream deep learning tasks. We do our best to keep it up to date. If you do find a task's SotA result is outdated or missing, please raise an issue (with: title of paper, dataset, metric, source code, and year). 

This summary is categorized into:

- [Computer Vision](#computer-vision)
- [Speech](#speech)
- [NLP](#nlp)
- [Contact](#contact)

## Computer Vision

### Classification

| Dataset  | Type         | Top-1 accuracy | Method                                  | Paper                                                                                                    | Code |
| -------- | ------------ | -------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---- |
| ImageNet | ResNet-50    | 78.35%         | ResNet-50 + DropBlock + label smoothing | [DropBlock: A Regularization Method for Convolutional Neural Networks](https://arxiv.org/abs/1810.12890) |      |
| ImageNet | Single model | 82.52%         | AmoebaNet-B + DropBlock                 | [DropBlock: A Regularization Method for Convolutional Neural Networks](https://arxiv.org/abs/1810.12890) |      |

### Object Detection

| Dataset                                                                | Type         | AP   | Method                     | Paper                                                                                          | Code |
| ---------------------------------------------------------------------- | ------------ | ---- | -------------------------- | ---------------------------------------------------------------------------------------------- | ---- |
| [MS-COCO 2017](http://cocodataset.org/index.htm#detection-leaderboard) | ResNet-101   | 43.4 | D-RFCN + SNIP + ResNet-101 | [An Analysis of Scale Invariance in Object Detection - SNIP](https://arxiv.org/abs/1711.08189) |      |
| [MS-COCO 2017](http://cocodataset.org/index.htm#detection-leaderboard) | Single model | 45.7 | D-RFCN + SNIP + DPN-98     | [An Analysis of Scale Invariance in Object Detection - SNIP](https://arxiv.org/abs/1711.08189) |      |

### Instance Segmentation

| Dataset                                                                | Type     | AP   | Method                    | Paper | Code                                                 |
| ---------------------------------------------------------------------- | -------- | ---- | ------------------------- | ----- | ---------------------------------------------------- |
| [MS-COCO 2018](http://cocodataset.org/index.htm#detection-leaderboard) | Ensemble | 48.6 | mmdet + FishNet, 5 models | -     | [PyTorch](https://github.com/open-mmlab/mmdetection) |

### Visual Question Answering

| Dataset                              | Type     | Score | Method | Paper                                                                                         | Code                                                  |
| ------------------------------------ | -------- | ----- | ------ | --------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [VQA](https://visualqa.org/roe.html) | Ensemble | 72.41 | Pythia | [Pythia v0.1: The Winning  Entry to the VQA Challenge 2018](https://arxiv.org/abs/1807.09956) | [PyTorch](https://github.com/facebookresearch/pythia) |

### Person Re-identification

| Dataset                                                                                                    | Type                    | Rank-1 accuracy | Method                                                                                | Paper                                                                                         | Code |
| ---------------------------------------------------------------------------------------------------------- | ----------------------- | --------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ---- |
| [Market-1501](http://www.liangzheng.org/Project/state_of_the_art_market1501.html)                          | Supervised single-query | 91.2%           | Pixel-level attention + region-level attention + joint feature learning               | [Harmonious Attention Network for Person Re-Identification](https://arxiv.org/abs/1802.08122) |      |
| [Market-1501](http://www.liangzheng.org/Project/state_of_the_art_market1501.html)                          | Supervised multi-query  | 93.8%           | Pixel-level attention + region-level attention + joint feature learning + multi-query | [Harmonious Attention Network for Person Re-Identification](https://arxiv.org/abs/1802.08122) |      |
| [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation/blob/master/State-of-the-art/README.md) | Supervised single-query | 85.95%          | SPReID                                                                                | [Human Semantic Parsing for Person Re-identification](https://arxiv.org/abs/1804.00216)       |      |

## NLP

### Language Modelling

| Dataset        | Type | Perplexity | Method | Paper                                                                                               | Code                                        |
| -------------- | ---- | ---------- | ------ | --------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Penn Tree Bank |      | 47.69      | MoS    | [Breaking the Softmax Bottleneck: A High-Rank RNN Language Model](https://arxiv.org/abs/1711.03953) | [PyTorch](https://github.com/zihangdai/mos) |
| WikiText-2     |      | 40.68      | MoS    | [Breaking the Softmax Bottleneck: A High-Rank RNN Language Model](https://arxiv.org/abs/1711.03953) | [PyTorch](https://github.com/zihangdai/mos) |

### Machine Translation

| Dataset                    | Type | BLEU | Method               | Paper                                                                                    | Code |
| -------------------------- | ---- | ---- | -------------------- | ---------------------------------------------------------------------------------------- | ---- |
| WMT 2014 English-to-French |      | 41.4 | Weighted Transformer | [Weighted Transformer Network for Machine Translation](https://arxiv.org/abs/1711.02132) |      |
| WMT 2014 English-to-German |      | 28.9 | Weighted Transformer | [Weighted Transformer Network for Machine Translation](https://arxiv.org/abs/1711.02132) |      |

### Text Classification

| Dataset | Type | Accuracy | Method | Paper                                                                        | Code                                                |
| ------- | ---- | -------- | ------ | ---------------------------------------------------------------------------- | --------------------------------------------------- |
| Yelp    |      | 68.6%    |        | [Learning Structured Text Representations](https://arxiv.org/abs/1705.09207) | [TensorFlow](https://github.com/nlpyang/structured) |

### Natural Language Inference

| Dataset                                                                               | Type     | Accuracy | Method | Paper                                                                                                                                                                                  | Code |
| ------------------------------------------------------------------------------------- | -------- | -------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/) | Single   | 89.9%    | GPT    | [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) |      |
| [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/) | Emsemble | 90.1%    |        | [Semantic Sentence Matching with Densely-Connected Recurrent and Co-Attentive Information](https://arxiv.org/abs/1805.11360)                                                           |      |
| [MultiNLI](https://www.kaggle.com/c/multinli-matched-open-evaluation/leaderboard)     | Emsemble | 86.7%    |        | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)                                                                   |      |

### Question Answering

| Dataset                                                  | Type         | F1     | Method     | Paper                                                                                                                | Code |
| -------------------------------------------------------- | ------------ | ------ | ---------- | -------------------------------------------------------------------------------------------------------------------- | ---- |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | Single model | 83.061 | BERT-large | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) |      |

### Named Entity Recognition

| Dataset    | Type         | F1   | Method     | Paper                                                                                                                | Code |
| ---------- | ------------ | ---- | ---------- | -------------------------------------------------------------------------------------------------------------------- | ---- |
| CoNLL-2003 | Single model | 92.8 | BERT-large | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) |      |

## Speech

### Acoustic Speech Recognition

| Dataset             | Type     | WER | Method                         | Paper                                                                                       | Code |
| ------------------- | -------- | --- | ------------------------------ | ------------------------------------------------------------------------------------------- | ---- |
| Switchboard Hub5'00 | Ensemble | 5.0 | biLSTM + CNN + Dense, 8 models | [The CAPIO 2017 Conversational Speech Recognition System](https://arxiv.org/abs/1801.00059) |      |

## Contact

Email: [cmsflash99@gmail.com](mailto:cmsflash99@gmail.com)
