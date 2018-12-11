# State-of-the-art results for deep learning tasks

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

| Dataset      | Type         | AP   | Method                     | Paper                                                                                          | Code |
| ------------ | ------------ | ---- | -------------------------- | ---------------------------------------------------------------------------------------------- | ---- |
| MS-COCO 2017 | ResNet-101   | 43.4 | D-RFCN + SNIP + ResNet-101 | [An Analysis of Scale Invariance in Object Detection - SNIP](https://arxiv.org/abs/1711.08189) |      |
| MS-COCO 2017 | Single model | 45.7 | D-RFCN + SNIP + DPN-98     | [An Analysis of Scale Invariance in Object Detection - SNIP](https://arxiv.org/abs/1711.08189) |      |

### Instance Segmentation

<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1703.06870.pdf'>Mask R-CNN</a></td>
      <td align="left"> <ul><li> COCO  </li></ul></td>
      <td align="left"> <ul><li> Average Precision: 37.1% </li></ul> </td>
      <td align="left"> <ul><li>  <a href='https://github.com/facebookresearch/Detectron'>Detectron (Official Version)</a> </li><li>  <a href='https://github.com/TuSimple/mx-maskrcnn'>MXNet</a> </li><li>  <a href='https://github.com/matterport/Mask_RCNN'>Keras</a> </li><li>  <a href='https://github.com/CharlesShang/FastMaskRCNN'>TensorFlow </a> </li></ul> </td>
      <td align="left">2017</td>    
    </tr>
  </tbody>
</table>

### Visual Question Answering

<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1708.02711'>Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge</a></td>
      <td align="left"> <ul><li> VQA  </li></ul></td>
      <td align="left"> <ul><li> Overall score: 69 </li></ul> </td>
      <td align="left"> <ul><li>   <a href=''>NOT FOUND</a> </li></ul> </li></ul> </td>
      <td align="left">2017</td>    
    </tr>
  </tbody>
</table>

### Person Re-identification

<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1708.04896'> Random Erasing Data Augmentation </a></td>
      <td align="left"> <ul><li> <a href='http://www.liangzheng.org/Project/project_reid.html'> Market-1501 </a> </li> <li> <a href='https://github.com/zhunzhong07/person-re-ranking'> CUHK03-new-protocol </a> </li> <li> <a href='https://github.com/layumi/DukeMTMC-reID_evaluation'> DukeMTMC-reID </a> </li> </ul></td>
      <td align="left"> <ul><li> Rank-1: 89.13 mAP: 83.93 </li>
          <li> Rank-1: 84.02 mAP: 78.28 </li>
          <li> labeled (Rank-1: 63.93 mAP: 65.05) detected (Rank-1: 64.43 mAP: 64.75) </li>
          </ul> </td>
      <td align="left"> <a href='https://github.com/zhunzhong07/Random-Erasing'> PyTorch </td>
      <td align="left">2017</td>    
    </tr>
  </tbody>
</table>

## NLP

### Language Modelling

<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
    </tr>  
    <tr>
      <td><a href='https://arxiv.org/pdf/1711.03953.pdf'>BREAKING THE SOFTMAX BOTTLENECK: A HIGH-RANK RNN LANGUAGE MODEL </a></td>
      <td align="left"><ul><li> PTB </li><li> WikiText-2 </li></ul></td>
      <td align="left"><ul><li> Perplexity: 47.69 </li><li> Perplexity: 40.68 </li></ul></td>
      <td align="left"><a href='https://github.com/zihangdai/mos'>PyTorch </a></td>
      <td align="left">2017</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1709.07432.pdf'>DYNAMIC EVALUATION OF NEURAL SEQUENCE MODELS </a></td>
      <td align="left"><ul><li> PTB </li><li> WikiText-2 </li></ul></td>
      <td align="left"><ul><li> Perplexity: 51.1 </li><li> Perplexity: 44.3 </li></ul></td>
      <td align="left"><a href='https://github.com/benkrause/dynamic-evaluation'>PyTorch </a></td>
      <td align="left">2017</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1708.02182.pdf'>Averaged Stochastic Gradient  Descent <br/> with Weight Dropped LSTM or QRNN </a></td>
      <td align="left"><ul><li> PTB </li><li> WikiText-2 </li></ul></td>
      <td align="left"><ul><li> Perplexity: 52.8 </li><li> Perplexity: 52.0 </li></ul></td>
      <td align="left"><a href='https://github.com/salesforce/awd-lstm-lm'>PyTorch </a></td>
      <td align="left">2017</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1711.00066.pdf'>FRATERNAL DROPOUT </a></td>
      <td align="left"><ul><li> PTB </li><li> WikiText-2 </li></ul></td>
      <td align="left"><ul><li> Perplexity: 56.8 </li><li> Perplexity: 64.1 </li></ul></td>
      <td align="left"> <a href='https://github.com/kondiz/fraternal-dropout'> PyTorch </a>  </td>
      <td align="left">2017</td>   
    </tr>
        <tr>
      <td><a href='https://arxiv.org/pdf/1703.10722.pdf'>Factorization tricks for LSTM networks </a></td>
      <td align="left">One Billion Word Benchmark</td>
      <td align="left"> Perplexity:  23.36</td>
      <td align="left"><a href='https://github.com/okuchaiev/f-lm'>Tensorflow </a></td>
      <td align="left">2017</td>   
    </tr>
  </tbody>
</table>

### Machine Translation

<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1711.02132.pdf'>WEIGHTED TRANSFORMER NETWORK FOR
MACHINE TRANSLATION</a></td>
      <td align="left"> <ul><li>WMT 2014 English-to-French </li><li>WMT 2014 English-to-German </li></ul></td>
      <td align="left"> <ul><li>  BLEU: 41.4 </li><li>   BLEU: 28.9 </li></ul> </td>
      <td align="left"> <ul><li><a href=''>NOT FOUND</a></li></ul></td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1706.03762'>Attention Is All You Need</a></td>
      <td align="left"> <ul><li>WMT 2014 English-to-French </li><li>WMT 2014 English-to-German </li></ul></td>
      <td align="left"> <ul><li>  BLEU: 41.0 </li><li>   BLEU: 28.4 </li></ul> </td>
      <td align="left"> <ul><li><a href='https://github.com/jadore801120/attention-is-all-you-need-PyTorch'>PyTorch</a> </li><li> <a href='https://github.com/tensorflow/tensor2tensor'>Tensorflow</a></li></ul></td>
      <td align="left">2017</td>    
    </tr>
     <tr>
      <td><a href='https://einstein.ai/static/images/pages/research/non-autoregressive-neural-mt.pdf'>NON-AUTOREGRESSIVE
NEURAL MACHINE TRANSLATION</a></td>
      <td align="left"> <ul><li> WMT16 Ro→En </li></ul></td>
      <td align="left"> <ul><li> BLEU: 31.44 </li></ul> </td>
      <td align="left"><ul><li><a href='https://github.com/salesforce/nonauto-nmt'>PyTorch</a></ul></li></td>
      <td align="left">2017</td>    
      </tr>
          <tr>
      <td><a href='https://arxiv.org/abs/1703.04887'> Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets</a></td>
      <td align="left"> <ul><li>NIST02    </li><li>NIST03 </li><li>NIST04 </li><li>NIST05 </li></ul></td>
      <td align="left"><li>38.74  </li><li>36.01  </li><li> 37.54 </li><li>33.76 </li></ul </td>
      <td align="left"> <ul><li><a href='https://github.com/ngohoanhkhoa/GAN-NMT'>NMTPY</a> </li></ul></td>
      <td align="left">2017</td>    
    </tr>
  </tbody>
</table>

### Text Classification

<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1705.09207'> Learning Structured Text Representations </a></td>
      <td align="left">Yelp</td>
      <td align="left">Accuracy: 68.6</td>
      <td align="left"> <ul><li><a href='https://github.com/nlpyang/structured'>Tensorflow</a></ul></li></td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1710.00519.pdf'>Attentive Convolution</a></td>
      <td align="left">Yelp</td>
      <td align="left">Accuracy: 67.36</td>
      <td align="left"> <ul><li><a href='https://github.com/yinwenpeng/Attentive_Convolution'>Theano</a></ul></li></td>
      <td align="left">2017</td>   
    </tr>
  </tbody>
</table>

### Natural Language Inference

| Dataset                                                                               | Type     | Accuracy | Paper                                                                                                                                                                                  | Code |
| ------------------------------------------------------------------------------------- | -------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/) | Single   | 89.9%    | [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) |      |
| [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/) | Emsemble | 90.1%    | [Semantic Sentence Matching with Densely-Connected Recurrent and Co-Attentive Information](https://arxiv.org/abs/1805.11360)                                                           |      |
| [MultiNLI](https://www.kaggle.com/c/multinli-matched-open-evaluation/leaderboard)     | Emsemble | 86.7%    | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)                                                                   |      |

### Question Answering

Leader Board

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)

<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1810.04805.pdf'> BERT-LARGE (ensemble) </a></td>
      <td align="left">The Stanford Question Answering Dataset</td>
      <td align="left"><ul><li> Exact Match: 87.4 </li><li> F1: 93.2 </li></ul></td>
      <td align="left"><ul><li><a href='https://github.com/google-research/bert'>Tensorflow</a></li><li><a href='https://github.com/huggingface/PyTorch-pretrained-BERT'>PyTorch</a> </td>
      <td align="left">2018</td>    
  </tr>
  </tbody>
</table>

### Named entity recognition

<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1710.11027.pdf'>Named Entity Recognition in Twitter using Images and Text </a></td>
      <td align="left">Ritter</td>
      <td align="left"><ul><li> F-measure: 0.59 </li></ul></td>
      <td align="left"><a href=''>NOT FOUND</a> </td>
      <td align="left">2017</td>    
  </tr>
  </tbody>
</table>

### Abstractive Summarization

| Research Paper                                                                                                                             | Datasets                                    | Metric                                                                                                                                                                                                                            | Source Code                                               | Year |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ---- |
| [Cutting-off redundant repeating generations </br> for neural abstractive summarization](https://aclanthology.info/pdf/E/E17/E17-2047.pdf) | <ul><li>DUC-2004</li><li>Gigaword</li></ul> | <ul><li>DUC-2004</li><ul><li> ROUGE-1: **32.28** </li><li> ROUGE-2: 10.54 </li><li>ROUGE-L: **27.80** </li></ul><li>Gigaword</li><ul><li> ROUGE-1: **36.30** </li><li> ROUGE-2: 17.31 </li><li>ROUGE-L: **33.88** </li></ul></ul> | NOT YET AVAILABLE                                         | 2017 |
| [Convolutional Sequence to Sequence](https://arxiv.org/pdf/1705.03122.pdf)                                                                 | <ul><li>DUC-2004</li><li>Gigaword</li></ul> | <ul><li>DUC-2004</li><ul><li> ROUGE-1: 33.44 </li><li> ROUGE-2: **10.84** </li><li>ROUGE-L: 26.90 </li></ul><li>Gigaword</li><ul><li> ROUGE-1: 35.88 </li><li> ROUGE-2: 27.48 </li><li>ROUGE-L: 33.29 </li></ul></ul>             | [PyTorch](https://github.com/facebookresearch/fairseq-py) | 2017 |

### Dependency Parsing

| Research Paper                                                                               | Datasets                                              | Metric                                                                | Source Code                                                                                         | Year                   |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------- |
| [Globally Normalized Transition-Based Neural Networks](https://arxiv.org/pdf/1603.06042.pdf) | <ul><li>Final CoNLL ’09 dependency parsing </li></ul> | <ul><li> 94.08% UAS accurancy</li> <li>92.15% LAS accurancy</li></ul> | <ul><li>[SyntaxNet](https://github.com/tensorflow/models/tree/master/research/syntaxnet) </li></ul> | <ul><li>2017</li></ul> |

## Speech

[Speech SOTA](https://github.com/syhw/wer_are_we)

### ASR

<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1708.06073.pdf'>The Microsoft 2017 Conversational Speech Recognition System</a></td>
      <td align="left"> <ul><li> Switchboard Hub5'00  </li></ul></td>
      <td align="left"> <ul><li> WER: 5.1  </li></ul> </td>
      <td align="left"> <ul><li>  <a href=''>NOT FOUND</a></li></ul> </td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1801.00059.pdf'>The CAPIO 2017 Conversational Speech Recognition System</a></td>
      <td align="left"> <ul><li> Switchboard Hub5'00  </li></ul></td>
      <td align="left"> <ul><li> WER: 5.0  </li></ul> </td>
      <td align="left"> <ul><li>  <a href=''>NOT FOUND</a></li></ul> </td>
      <td align="left">2017</td>    
    </tr>
  </tbody>
</table>

## Contact

Email: cmsflash99@gmail.com
