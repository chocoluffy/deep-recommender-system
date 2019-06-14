# Multi-Interest Network with Dynamic Routing for Recommendation at Tmall 

评分：5/5。  
简介：引入capsule的dynamic routing和label-aware attention，对用户历史行为序列（点击商品的集合）提取用户兴趣特征（user embedding）。相比阿里之前的DIN在用户行为聚类上更进了一步，本质上是商品特征的soft clustering，并根据电商环境进行了改良。  

- 传统方式CF对稀疏数据集表现不好，而且存在MF计算量大的问题。而DNN的方式的局限在将用户特征用一个低维的向量来表示，对于多兴趣方向行为的用户损失了信息。DIN采用self attention的做法，使得对每一个目标商品可以attend to globel items，并在最后通过sigmoid预测CTR，速度慢，更适合Ranking精排序阶段。而MIND的主要任务和DIN不同，MIND主要负责输出用户嵌入向量，由于user embedding和item embedding在同一个vector space，可以快速通过内积的nearest neighbor得到大致推荐商品范围（粗排在一千个左右），为Matching粗排序阶段的筛选作准备。
- 在dynamic routing输出了兴趣聚类capsules之后，接label-aware attetion层可以让训练让target item表示为interest capsules的一个线性组合。其实Key为target item，Q和V为interest capsules。
- 具体用户兴趣聚类的个数可以动态调整。比如是用户历史行为的log。由于推荐系统环境中用户行为序列的长度不定（variable length），改进dynamic routing采用fixed shared weight使得未知商品同样能够作为网络输入。
- 类似Airbnb real-time personalization的做法。目前几种主流将匹配机制做到实时的方法本质上都是将user和item embedding训练在同一个vector space，然后得以通过内积来进行快速比较。同时为了反映用户行为的变化，输入user embedding的架构需要对variable length的行为序列兼容，或者采取most recent N的固定架构。
- 和其他Matching粗排序阶段的算法比较发现，MIND > item-based CF > Youtube DNN.

# highlights

- Most of the existing deep learning-based models represent one user as a single vector which is insufficient to capture the varying nature of user’s interests

- Specifically, we design a multi-interest extractor layer based on capsule routing mechanism, which is applicable for clustering historical behaviors and extracting diverse interests.  Furthermore, we develop a technique named label-aware attention to help learn a user representation with multiple vectors

- . For both of the two stages, it is vital to model user interests and find user representations capturing user interests, in order to support efficient retrieval of items that satisfy users’ interests.

- Collaborative filtering-based methods represent user interests by historical interacted items [22] or hidden factors [17], which suffer from sparsity problem or computationally demanding.

- [17] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 8 (2009), 30–37.

- [22] Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2001. Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th international conference on World Wide Web. ACM, 285–295.

- Deep learning-based methods usually represent user interests with low-dimensional embedding vectors. For example, the deep neural network proposed for YouTube video recommendation (YouTube DNN) [7] represents each user by one fixed-length vector transformed from the past behaviors of users, which can be a bottleneck for modeling diverse interests, as its dimensionality must be large in order to express the huge number of interest profiles at Tmall

- Nevertheless, the adoption of attention mechanisms also makes it computationally prohibitive for large-scale applications with billion-scale items as it requires re-calculation of user representation for each item, making DIN only applicable for the ranking stage.

- [7] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 191–198.

- In this paper, we focus on the problem of modeling diverse interests of users in the matching stage.

- [21] Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. 2017. Dynamic routing between capsules. In Advances in Neural Information Processing Systems. 3856–3866

- The process of dynamic routing can be viewed as soft-clustering, which groups userâĂŹs historical behaviors into several clusters. Each cluster of historical behaviors is further used to infer the user representation vector corresponding to one particular interest.

- The user representation vectors are computed only once and can be used in the matching stage for retrieving relevant items from billion-scale items.

- [3] Zeynep Batmaz, Ali Yurekli, Alper Bilge, and Cihan Kaleli. 2018. A review on deep learning for recommender systems: challenges and remedies. Artificial Intelligence Review (2018), 1–37

- 31] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 1059–1068.

- 23] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. ACM, 565–573.

- [14] Geoffrey E Hinton, Sara Sabour, and Nicholas Frosst. 2018. Matrix capsules with EM routing. In International Conference on Learning Representations.

- These two main differences to conventional neural network make capsule networks capable of encoding the relationship between the part and the whole,

- During training, an extra label-aware attention layer is introduced to guide the training process.

- a tuple (Iu, Pu, Fi), where Iudenotes the set of items interacted by user u (also called user behavior), Puthe basic profiles of user u (like user gender and age), Fithe features of target item (such as item id and category id).

- The core task of MIND is to learn a function for mapping raw features into user representations, which can be formulated as Vu= fuser(Iu, Pu) , (1) where Vu= ?− →v1 u, ...,− →vK u ? ∈ Rd×Kdenotes the representation vectors of user u, d the dimensionality, K the number of representation vectors. When K = 1, one representation vector is used, just like YouTube DNN.

- Besides, the representation vector of target item i is obtained by an embedding function as − →ei= fitem(Fi) , (2) where− →ei∈ Rd×1denotes the representation vector of item i, and the detail of fitemwill be illustrated in the "Embedding & Pooling Layer" section.

- Each group contains several categorical id features, and these id features are of extremely high dimensionality. For instance, the number of item ids is about billions, thus we adopt the widely-used embedding technique to embed these id features into low-dimensional dense vectors (a.k.a embeddings), which significantly reduces the number of parameters and eases the learning process.

- For item ids along with other categorical ids (brand id, shop id, etc.) that have been proved to be useful for cold-start items [25] from Fi, corresponding embeddings are further passed through an average pooling layer to form the label item embedding− →ei.

- 25] Jizhe Wang, Pipei Huang, Huan Zhao, Zhibo Zhang, Binqiang Zhao, and Dik Lun Lee. 2018. Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery &#38; Data Mining (KDD ’18). 839–848.

- 13] Geoffrey E Hinton, Alex Krizhevsky, and Sida D Wang. 2011. Transforming auto-encoders. In International Conference on Artificial Neural Networks. Springer, 44–51.

- the routing logit bijbetween low-level capsule i and highlevel capsule j is computed by bij= (− →ch j)TSij− →cl i, (4) where Sij ∈ RNh×Nldenotes the bilinear mapping matrix to be learned.

- On the one hand, user behaviors are of variable-length, ranging from dozens to hundreds for Tmall users, thus the use of fixed bilinear mapping matrix is generalizable.  On the other hand, we hope interest capsules lie in the same vector space, but different bilinear mapping matrice would map interest capsules into different vector spaces. Thus, the routing logit is calculated by bij=− →uT jS− →ei, i ∈ Iu, j ∈ {1, ..., K},

- we introduce a heuristic rule for adaptively adjusting the value of K for different users.  Specifically, the value of K for user u is computed by K′u= max(1, min(K, log2(|Iu|))).

- Therefore, during training, we design a label-aware attention layer based on scaled dot-product attention [24] to make the target item choose which interest capsule is used. Specifically, for one target item, we calculate the compatibilities between each interest

- and compute a weighted sum of interest capsules as user representation vector for the target item, where the weight for one interest capsule is determined by corresponding compatibility

- In label-aware attention, the label is the query and the interest capsules are both keys and values,

- where pow denotes element-wise exponentiation operator, p a tunable parameter for adjusting the attention distribution. When p is close to 0, each interest capsule attends to receive even attention. When p is bigger than 1, as p increases, the value has bigger dot-product will receive more and more weight.

- With the user vector− →vuand the label item embedding− →eiready, we compute the probability of the user u interacting with the label item i as Pr(i|u) = Pr ?− →ei|− →vu ? = exp ?− →vT u− →ei ? ? j ∈Iexp ?− →vT u− →ej ? .  (10) Then, the overall objective function for training MIND is L = ? (u,i)∈D log Pr(i|u)

- Thus, we use the sampled softmax technique [7] to make the objective function trackable and choose the Adam optimizer [16] for training MIND.

- At serving time, user’s behavior sequence and user profile are fed into the fuserfunction, producing multiple representation vectors for each user.

- DIN applies an attention mechanism at the item level, while MIND employs dynamic routing to generate interest capsules and considers diversity at the interest level

- 26] Jason Weston, Ron J Weiss, and Hector Yee. 2013. Nonlinear latent factorization by embedding multiple user interests. In Proceedings of the 7th ACM conference on Recommender systems. ACM, 65–68.

- The improvement of MIND1-interest over YouTube DNN shows that dynamic routing serves as a better pooling strategy than average pooling.

- 2) Label-aware attention layer makes target item attend over multiple user representation vectors, enabling more accurate matching between user interests and target item.

- There are two baseline methods for online experiments. One is item-based CF, which is the base matching algorithm serving the majority of the online traffic. The other is YouTube DNN, which is the well-known deep learning-based matching model. We deploy all comparing methods in an A/B test framework, and one thousand of candidate items are retrieved by each method, which then fed to the ranking stage for final recommendation.

- The whole procedure of selecting thousands of candidate items from the billion-scale item pool by User Interest Extractor 8 and Recall Engine can be fulfilled in less than 15 milliseconds, due to the effectiveness of serving based on MIND. Taking a tradeoff between the scope of items and the response time of the system, top 1000 of these candidate items are scored by Ranking Service which predicts CTRs with a bunch of features.

- For future work, we will pursue two directions. The first is to incorporate more information about user’s behavior sequence, such as behavior time etc. The second direction is to optimize the initialization scheme of dynamic routing, referring to K-means++ initialization scheme, so as to achieve a better user representation

