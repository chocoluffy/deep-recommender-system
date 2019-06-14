# 目录

## Deep Learning

- [Softmax的numpy实现, 以及SGD、minibatch](https://github.com/chocoluffy/deep-learning-notes/blob/master/DL/Softmax.ipynb)
- [Ridge Regression的实现](https://github.com/chocoluffy/deep-learning-notes/blob/master/DL/Ridge%20Regression.ipynb)
- [Customized Convolution Neural Network的Pytorch实现，包括batch normalization](https://github.com/chocoluffy/deep-learning-notes/blob/master/DL/CNN_pytorch.ipynb)
- [GAN on Fashion-MNIST dataset，Pytorch实现](https://github.com/chocoluffy/deep-learning-notes/blob/master/DL/GAN.py)

## Kaggle

- [Kaggle Jupyter技巧总结：经典机器学习模型，Pipeline, GridSearch, Ensemble等](https://github.com/chocoluffy/kaggle-notes/tree/master/Kaggle)


## Recommendation System

- 【5/5】[Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://github.com/chocoluffy/deep-recommender-system/tree/master/RecSys/MIND)
- 【4/5】[BERT4Rec- Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://github.com/chocoluffy/deep-recommender-system/tree/master/RecSys/BERT4Rec)
- 【3/5】[Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://github.com/chocoluffy/deep-recommender-system/tree/master/RecSys/Transformer-in-WDL)
- 【5+/5】[Deep Neural Networks for YouTube Recommendations](https://github.com/chocoluffy/deep-learning-notes/tree/master/RecSys/Youtube-DNN)
- 【4/5】[Collaborative Deep Learning for Recommender Systems](https://github.com/chocoluffy/deep-learning-notes/tree/master/RecSys/Collaborative-Deep-Learning)
- 【4/5】[Wide & Deep Learning for Recommender Systems](https://github.com/chocoluffy/deep-learning-notes/tree/master/RecSys/Wide%26Deep)
- 【5/5】[Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://github.com/chocoluffy/deep-learning-notes/tree/master/RecSys/Embedding-Airbnb)
- 【3/5】[A Cross-Domain Recommendation Mechanism for Cold-Start Users Based on Partial Least Squares Regression](https://github.com/chocoluffy/deep-learning-notes/tree/master/RecSys/PLSR) 
- 【5/5】[IRGAN - A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models](https://github.com/chocoluffy/deep-learning-notes/tree/master/RecSys/IRGAN) 
- 【4/5】[Practical Lessons from Predicting Clicks on Ads at Facebook](https://github.com/chocoluffy/kaggle-notes/tree/master/RecSys/predicting-clicks-facebook)


### 概述

### [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://github.com/chocoluffy/deep-recommender-system/tree/master/RecSys/MIND) 

评分：5/5。  
简介：引入capsule的dynamic routing和label-aware attention，对用户历史行为序列（点击商品的集合）提取用户兴趣特征（user embedding）。相比阿里之前的DIN在用户行为聚类上更进了一步，本质上是商品特征的soft clustering，并根据电商环境进行了改良。  

- 传统方式CF对稀疏数据集表现不好，而且存在MF计算量大的问题。而DNN的方式的局限在将用户特征用一个低维的向量来表示，对于多兴趣方向行为的用户损失了信息。DIN采用self attention的做法，使得对每一个目标商品可以attend to globel items，并在最后通过sigmoid预测CTR，速度慢，更适合Ranking精排序阶段。而MIND的主要任务和DIN不同，MIND主要负责输出用户嵌入向量，由于user embedding和item embedding在同一个vector space，可以快速通过内积的nearest neighbor得到大致推荐商品范围（粗排在一千个左右），为Matching粗排序阶段的筛选作准备。
- 在dynamic routing输出了兴趣聚类capsules之后，接label-aware attetion层可以让训练让target item表示为interest capsules的一个线性组合。其实Key为target item，Q和V为interest capsules。
- 具体用户兴趣聚类的个数可以动态调整。比如是用户历史行为的log。由于推荐系统环境中用户行为序列的长度不定（variable length），改进dynamic routing采用fixed shared weight使得未知商品同样能够作为网络输入。
- 类似Airbnb real-time personalization的做法。目前几种主流将匹配机制做到实时的方法本质上都是将user和item embedding训练在同一个vector space，然后得以通过内积来进行快速比较。同时为了反映用户行为的变化，输入user embedding的架构需要对variable length的行为序列兼容，或者采取most recent N的固定架构。
- 和其他Matching粗排序阶段的算法比较发现，MIND > item-based CF > Youtube DNN.


### [BERT4Rec- Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://github.com/chocoluffy/deep-recommender-system/tree/master/RecSys/BERT4Rec)

评分：4/5。  
简介：将Bert双向Transformer的结构带入了推荐系统，并改变了目标用Cloze task来防止信息泄漏并可用于预测随机masked的item。  

- 其中一个关键的假设：用户行为序列并没有NLP那样严格的逻辑依赖关系，更多在视频推荐场景里强调的是相关性，发现性和多样性。在Transformer里依旧是单向的次序结构，只不过相比RNN来说，Transformer能够在每一步都连接过去所有的input，而不必将信息condense到一个hidden state里（信息噪音和流失）。
- Transformer的引入相比RNN对并行计算友好。复杂度从O(nd^2)变为O(n^2d)，其中n为序列长度而d为特征长度，对于短序列高维的特征表示来说，self-attention是一个选择。同时由于只引入了matrix multiplication的操作，对SGD运算友好。
- self attention本身其实和位置无关，只是一种扩大receptive field的方式，类似CNN中利用叠加起来的层次来一部部增大感受域，如果需要次序的概念，需要加入position encoding。
- 在推荐系统里，position embedding对应的意义是此操作距离用户当前时刻的时间差。这个position embedding在不同的场景应当有不同的意义和改进技巧。比如在图片领域强调的是translation invariance那么相比absolute position更合适的其实是relative position encoding等等。

### [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://github.com/chocoluffy/deep-recommender-system/tree/master/RecSys/Transformer-in-WDL)

评分：3/5。  
简介：将Transformer的self attention结构应用在推荐系统典型的Wide & Deep网络结构中。  

- 结合了position embedding，用距离当前推荐时间的时间差作为位置信息。
- 采用的是内部的attention机制，也即Q = K = V = embedding，其中dot product计算的是物品之间的相似程度。最后采用multi-head的做法，类比CNN中使用的多个kernel得到多个feature map，multi-head使得能够探索出embedding不同位置的特性。注意的一点是，attention同样可以引入外部的embedding，只要保证key和value是一一对应的即可，可以利用外部embedding来升、降维度。
- 最终的目标是预测目标产品的CTR的概率，适合电商的环境。对于视频推荐的场景，可以尝试youtube那篇文章的目标，即expected watch time。

### [Deep Neural Networks for YouTube Recommendations](https://github.com/chocoluffy/deep-learning-notes/tree/master/RecSys/Youtube-DNN)

评分：5+/5。  
简介：使用DNN对大规模线上推荐系统架构的一次综述，包含Candidate Generation和Ranking两部分。Candidiate Gneration的部分负责生成user embedding，借鉴wordvec的skip gram negative sampling模型，Ranking部分使用类似的架构，并用weighted LR将目标改为预计观看时间。很经典的文章。  

- DNN的方法可以看作是泛化的matrix factorization的类别，但是优势在于可以加入任意连续或者类别的特征，适合不断迭代。
- 在线系统获得的显式特征较稀疏，比如点赞、收藏等用户行为。训练时更多利用到的是隐式特征，比如用户观看市场，是否完整观看等等。
- CF中并没有强调次序的观念，而youtube的DNN做法则只选取在用户观看视频之前的N个操作数据作为输入。同时使用unordered bag来防止次序带来的直接影响。比如推荐用户刚搜索过的视频等。
- 鉴于用户点击视频事件highly unbalanced的属性，引入以用户观看时长为权重的weighted logistic regression，并推导可用odds来近似用户观看视频时长的预期（值域相同）。
- 线上serving的场景中，可以根据用户的最新操作实时更新user embedding，而video embedding则是SGNS训练中的副产物，需要定期全量重训。实时场景里将embedding存入内存用ANN的方法，比如LSH，来推荐视频，而不需要对全量视频跑inference。
- example age特征的引入很关键，指该用户操作距离此训练的时间间隔，使得模型可以模拟实际视频走红的时序特征。

### [Collaborative Deep Learning for Recommender Systems](https://github.com/chocoluffy/deep-learning-notes/tree/master/RecSys/Collaborative-Deep-Learning)

评分：4/5。  
简介：针对rating和content information matrix，设计MAP(Maximum A Priori)的objective function来改善user embedding。相比传统collaborative filtering不擅长直接处理稀疏rating输入，CDL通过更好地结合content information可以得到更好的rating prediction。  

- 引入SDAE(Stacked Denoising Auto Encoder)来获得item的compact feature。随机初始化单位user embedding，长度等同于SDAE中encoder的输出维度，使得`uTv = r`为目标来训练。其中v为encoder输出，及item的compact feature。构造MAP的objective利用EM交替更新参数。
- 参数初始化都用bayesian model。整体上是hierachial bayesian model。
- 使用recall作为evaluation metric，即在推荐结果中relevant的数量占全部relevant item的数量。对比precision。

### [Wide & Deep Learning for Recommender Systems](https://github.com/chocoluffy/deep-learning-notes/tree/master/RecSys/Wide%26Deep)

评分：4/5。  
简介：利用logistic regression针对广度的交叉特征(cross product transformation)，利用NN负责深度特征挖掘，并同时进行joint training。来自Google的工程实践总结。  

- 提出Wide & Deep的解决方案来改善memorization(relevancy)和generality(diversity)的表现。改善传统embedding容易因为稀疏输入而over-generalize的问题。
- 介绍了工业上推荐系统的流程，先通过retrival从数据库选出初步的candidate，O(100)的量级；然后再通过rank的模型将candidate进行精排返回前十作为结果，Wide & Deep是在这个rank阶段的一种方案。
- 每一次retrain时利用上一次的weight初始化，以减少训练时间，类比transfer learning。
- 为了线上的低延迟在10ms量级，采取多线程small batch的方式跑inference，而不是将所有candidate放在同一个batch里跑。最终batch size为50，4个线程可以达到14ms的表现。
- Matrix Factorization里通过dot product引入interaction的特征，其根本目的是为了引入non-linearity。但这部分可以用NN更好的完成。

### [Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://github.com/chocoluffy/deep-learning-notes/tree/master/RecSys/Embedding-Airbnb)

评分：5/5。  
简介：对listing做embedding，将user type，listing type以及query在同一个vector space构建embedding，以实时更新搜索结果并提高准度。KDD 2018 best paper。  

- 利用in-session signal，将用户行为（包括点击、最终达成交易以及被拒绝）模拟为一个时序序列，类比word2vec中的单个句子。然后利用skip-gram和negative sampling来进行word2vec模型的训练。
- 对实际场景的精准观察是很多机制设计的灵感来源。比如：
    - 将最终的bookings作为global context添加进每一个sliding window的训练。
    - 因为旅游目的地的搜索是congregated search，于是添加在target area区域内的negative sampling sets。
    - 将奖赏(vetor距离更新)和惩罚(vector相互远离)加入进objective function的设计。
    - 考虑到用户booking的稀疏性，采用user type(users' many-to-one relation)，并将listing type和query type并入同一序列进行训练，使得embedding在同一个vector space，以允许用户在搜索时可以提供语义上最接近的结果(而不是简单匹配)，并改善最相近listing carousel模块的推荐结果。

### [A Cross-Domain Recommendation Mechanism for Cold-Start Users Based on Partial Least Squares Regression](https://github.com/chocoluffy/deep-learning-notes/tree/master/RecSys/PLSR)

评分：3/5。  
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

评分：4/5。  
简介：Facebook提出的CTR预估模型，GBDT + Logistic Regression。

- 加深了对entropy的理解，以及在CTR领域使用normalized entropy的实践。
- 学习到了GBM和LR的结合。用boosted decision tree来主要负责supervised feature learning有很大的优势。之后对接的LR + SGD可以作为online learning保持日常更新训练保证data freshness。
- 对引入prior的理解。[1] integrating knowledge. [2] avoid overfitting, similar to regularization.
- 在online learning领域，对比LR + SGD和BOPR（Bayesian online learning scheme for probit regression）的优劣势。