This repo contains some exercises and information for deep learning tutorials for the University of Chicago's Data Science Institute.

Note: The code in this repo is for some of the later lessons in this sequence. There is a `main` branch which is a skeleton of a training loop for a computer vision model, a `working` branch which includes the code from the computer vision model, and a `bugs` branch which includes several common problems that either break model training or seriously impact model training results.

Our deep learning tutorials are focused on developing some theoretical intuition for what deep learning models are doing as well as hands-on PyTorch skills.

Here is the outline for the tutorials as well as links to activity notebooks and additional resources.

### Lesson 1: Linear Regression & Parameter Estimation

Parameter estimation using least squares and gradient descent: https://colab.research.google.com/drive/10yrQPaX80lKSBeuwkG9NhrBiloyr6Pco?usp=sharing

### Lesson 2: Logistic Regression

Logistic regression using PyTorch: https://colab.research.google.com/drive/1g-VKPk7guUFui7MkTZenISBzUICp6dF5?usp=sharing

### Lesson 3: Computer Vision & Convolutional Neural Networks

Image classification using a custom PyTorch model and a pretrained model: https://colab.research.google.com/drive/1qc5WJqUkvkidt92Q16E4gYVK4d0QEPl5?usp=sharing

Note: This is a modified version of this official PyTorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

#### Resources
- Stanford's CS231n: Convolutional Neural Networks for Visual Recognition: https://cs231n.github.io/convolutional-networks/
  - There's an animation of the convolution operation about halfway down the page that is very helpful

### Lesson 4: Cross-Entropy Loss & Back Propagation

There was no specific notebook for this lesson. We reviewed the computer vision notebook and spent more time talking about cross-entropy loss, how PyTorch implements loss functions, and how back propagation works.

#### Activity

Implement the code from the computer vision tutorial notebook using the `main` branch of this repo as a base.

### Lesson 5: Transformers & Natural Language Processing

Text classification using Huggingface: https://colab.research.google.com/drive/1IXcGDt83mhtkNO13rFNDvIh3zSCo2sIV?usp=sharing

Note: This is a modified version of this official Huggingface tutorial: https://huggingface.co/learn/nlp-course/en/chapter3/4?fw=pt

#### Resources
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- Visualizing Attention: https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
- Attention is all you need: https://arxiv.org/abs/1706.03762

### Lesson 6: Debugging

Fix the code in the `bugs` version of this repo. See if you can get the same accuracy as the `working` version of this repo.
