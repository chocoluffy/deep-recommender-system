# 目录

## Deep Learning

- [Softmax的numpy实现, 以及SGD、minibatch](https://github.com/chocoluffy/deep-learning-notes/blob/master/DL/Softmax.ipynb)
- [Ridge Regression的实现](https://github.com/chocoluffy/deep-learning-notes/blob/master/DL/Ridge%20Regression.ipynb)
- [Customized Convolution Neural Network的Pytorch实现，包括batch normalization](https://github.com/chocoluffy/deep-learning-notes/blob/master/DL/CNN_pytorch.ipynb)

## Kaggle

- [Kaggle Jupyter技巧总结：经典机器学习模型，Pipeline, GridSearch, Ensemble等](https://github.com/chocoluffy/kaggle-notes/tree/master/Kaggle)


## Recommendation System

### [IRGAN - A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models](https://github.com/chocoluffy/deep-learning-notes/tree/master/RecSys/IRGAN)

评分：5/5。
简介：将GAN应用在information retrieval上。SIGIR2017满分论文。

- 巧妙地构造了一个minmax的机制。discriminator负责判断一个document是否well-matched，通过maximum likelihood。而对generator来说，则是更新参数minimize这个maximum likelihood。可以借鉴的点，在于如何设计的likelihood的期望。Discriminator其实核心就是一个binary classifier，然后利用logistic转换到(0, 1)的值域范围，就可以设计`log(D(d|q)) + log(1-D(d'|q))`的likelihood来达到目标！（其中d'为generator的样本，d为ground truth distribution的样本）。理解的思路其实很简单，generator生成的d'是试图欺骗discriminator的，因此如果D判定d'为well-matched，则因此可以引入large loss来penalize discriminator，也是`log(1-D(d'|q))`的设计思路。
- 相比传统的GAN在continuous latent space生成图片，IRGAN的generator则是在document pool选择最相关的relevant document，是离散的。于是引入RI里的policy graident来descent。
- 利用hierachial softmax来降低softmax的复杂度。传统的softmax和所有的document相关，而hierachial softmax可以降至log(|D|).
- MLE(Maximum Likelihood Estimation)是MAP(Maximum A Priori)的一种特殊情况，即Prior为uniform distribution的。MAP既考虑了likelihood还考虑了参数的Prior。

### [Practical Lessons from Predicting Clicks on Ads at Facebook](https://github.com/chocoluffy/kaggle-notes/tree/master/RecSys/predicting-clicks-facebook)

评分：5/5。  
简介：Facebook提出的CTR预估模型，GBDT + Logistic Regression。

- 加深了对entropy的理解，以及在CTR领域使用normalized entropy的实践。
- 学习到了GBM和LR的结合。用boosted decision tree来主要负责supervised feature learning有很大的优势。之后对接的LR + SGD可以作为online learning保持日常更新训练保证data freshness。
- 对引入prior的理解。[1] integrating knowledge. [2] avoid overfitting, similar to regularization.
- 在online learning领域，对比LR + SGD和BOPR（Bayesian online learning scheme for probit regression）的优劣势。