# 目录

## Deep Learning

- [Softmax的numpy实现, 以及SGD、minibatch](https://github.com/chocoluffy/deep-learning-notes/blob/master/DL/Softmax.ipynb)
- [Ridge Regression的实现](https://github.com/chocoluffy/deep-learning-notes/blob/master/DL/Ridge%20Regression.ipynb)
- [Customized Convolution Neural Network的Pytorch实现，包括batch normalization](https://github.com/chocoluffy/deep-learning-notes/blob/master/DL/CNN_pytorch.ipynb)

## Kaggle

- [Kaggle Jupyter技巧总结：经典机器学习模型，Pipeline, GridSearch, Ensemble等](https://github.com/chocoluffy/kaggle-notes/tree/master/Kaggle)


## Recommendation System

### A Cross-Domain Recommendation Mechanism for Cold-Start Users Based on Partial Least Squares Regression

评分：3.5/5。
简介：利用PLSR来解决用户推荐场景里cold start的问题。

- PLSR适合针对多模态的数据特征进行回归拟合，原理是在压缩降维时考虑最大化cross-domain数据的covariance，区别于PCA，LSI等仅仅最大化单个domain数据的variance。论文针对用户在没有任何历史评分记录的target domain中，利用已有的可能完全不同种类的source domain的评分记录来进行预测。
- PLSR的一个前提是需要完整的matrix信息，而user rating matrix的训练数据往往稀疏而且有很多缺失的entry。常见的预处理是利用MF(Matrix Factorization)或其变式NMF(Non-negative MF)来填充空缺的entry。
相比SVD，MF能够利用Alternating Least Square的方式并行化运算较快，但是并没有能够像SVD中那样可以选择合适的top K rank.
- 论文中提到一个生产环境中加速的技巧，针对user rating matrix往往维度过大导致较高的计算复杂度，是在对数据预处理(补值)之后再进行一次MF来拿到user latent factor(n*k)，然后利用这个数据在往后的计算中代替原始的user rating matrix以降低计算复杂度。**online阶段加入新用户时再一次的MF计算，可以利用incremental MF的技巧来避免重复计算。**
- 其他类似的推荐算法：
    - crossMF：合并training，test数据一并使用MF来预测值。
    - crossCBMF(clustering-based MF)：state-of-art。
- 本质上PLSR也属于linear regression的一种，对于模拟数据间非线性的关系可以考虑kernel PLSR。对于传统机器学习算法来说，很多时候从线性到非线性的处理基本都会借助kernel来直接代替曾经的点乘操作，比如SVM。
- 联想到我曾在一个reverse image search engine的kaggle竞赛中用到PLSR针对image feature vector和word embedding进行跨模态建模，效果不错，最终LB拿了第一名。

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