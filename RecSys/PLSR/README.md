# A Cross-Domain Recommendation Mechanism for Cold-Start Users Based on Partial Least Squares Regression

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

# highlights

- The major advantages are being able to address the cold-start problem and to deal with the data sparsity problem in the rating data of the target domain

- Specifi- cally, in cross-domain recommendation, the rating data can be divided into source domain and tar- get domain. The former provides the knowledge about the user-rating preferences, while the user rating scores on items in the latter are what we aim to infer. Since the target domains are usually presumed to suffer from insufficient ratings (especially on new users) that lead to the user cold- start problem and the rating sparsity problem, one of the well-known and effective solutions is to transfer the knowledge of rating preferences from the source domain that possesses rich ratings to remedy these problems

- They either resort to the contextual information of ratings (e.g., the social network behind users, the tags associated with users and items, and the user profiles) or presume that new users possess a few ratings before the recommendation is performed.

- We propose the PLSR-CrossRec model.

The central idea is to exploit the users who had ever produced ratings in both source and target domain, to learn the rating knowledge. The knowledge to be learned and transferred is the set of regression coefficients, which relates source-domain ratings to target-domain ratings

- PLSR-Latent aims at predicting the latent factors of users in the target domain, instead of directly predicting the target-domain rating scores.

- Because the dimensionality of latent factors is much lower than the original rating matrices, the computational complexity at the training stage can be significantly lowered down.

- Third, PLSR-Latent can stably afford sparse source user-rating matrices with satisfying performance. Fourth, the time efficiency of PLST-Latent at the training stage is promising.

- There are three most well-known approaches, content-based, collaborative filtering, and hybrid [1, 3]. Content-based recommen- dation encodes user profiles and items’ textual contents as features, then compute the similarity between users and items in the feature space for recommendation [20]. Collaborative filtering takes advantage of the preferences between users based on the user-rating matrix. It can be di- vided into three solution types, user-based [4], item-based [27], and latent factor [13]. The latent factor approach had been validated to be widely promising. Its idea is to perform matrix factor- ization to find the latent factor for each user, in which each of the reduced dimensions implies a certain hidden concept. Similarity computation is applied in the space of latent factors to find the items that are most similar to the user’s preference.

- Phase 1: Matrix Completion. For the missing values in both matrices of the source and the target domain, we plan to exploit the technique of Matrix Factorization (MF) [13] to initialize the unknown rating scores so that the rating matrices X and Y can be completed.

- Specifically, given two incomparable variables, the PLS technique is able to transform both variables into a low-dimensional space so that they can com- pare with one another and their similarity can be derived

- By converting music and movies into a single low-dimensional space via some feature transformation powered by PLS, one is allowed to match movies with music songs.

- 

- In other words, by the linear transformation with r and s, each user’s ratings can be combined into a certain numerical value,

- Through Equation (3), the derived latent factors can reflect the dimensions with higher covariance and higher correlation between domains, and thus the most useful knowledge can be kept and used to predict the target-domain ratings of cold-start users. However, although conventional dimension reduction techniques, such as Principle Com- ponent Analysis [12] and Latent Semantic Indexing [5], can be used to learn the most informative dimensions from rating data, the learned latent factors can only belong to separate rating domain, instead of those shared by both source and target domains. Hence, such methods are not suitable to be directly applied for cross-domain recommendation.

- The central idea is exploit- ing the matrices of latent factors in both source and target domains to learn the matrix of regres- sion coefficient in the PLS regression. Specifically, instead of using the completed rating matrices X and Y, whose dimensions are too large, we take advantage of the derived latent factors from matrix factorization, denoted by PXand PY, to compute the regression coefficients B. The latent factor matrices, PXand PY, can be obtained from matrix factorization,

- CrossMF. Matrix factorization (MF) [13] is one of the popular collaborative filtering tech- niques for single-domain recommender systems. The idea of MF is to learn the latent factors of users and items from training data, and accordingly make the rating prediction. To extend the conventional MF technique for cross-domain recommendation, we combine both train- ing and testing data, and combine both source and target domains. An unified cross-domain user-rating matrix is then constructed. By factorizing this unified matrix, we can derive the latent factors, then accordingly obtain the complete matrix, in which new users’ ratings in the target domain can be derived. We call this method CrossMF. Since we empirically find that the number of dimensions K = 5 leads to the best performance in a variety of domain pairs, we choose to set K = 5 in CrossMF for all of the experimental comparison.

- CrossNMF. Non-negative matrix factorization (NMF) [14] is one of the variants of matrix factorization techniques. NMF adds a constraint when factorizing the matrix. The constraint is enforcing all the values in the latent factors to be non-zero values. The reason of making such a constraint is that negative values in the latent factors cannot preserve the physical meaning of each dimension, i.e., difficult to be explained. We employ the strategy similar with CrossMF to modify NMF for cross-domain recommendation. That said, we apply NMF to factorize an unified matrix combining training and testing data and source- and target- domain data. We call this method CrossNMF. Likewise, we set the number of dimensions K = 5, since it empirically leads to the best performance.

- CrossCBMF. Clustering-based matrix factorization (CrossCBMF) [19] is the state-of-the- art method that purely considers user-item rating information for cross-domain recom- mendation and for cold-start users in the target domain. The central idea of CrossCBMF is leveraging the partially overlapped users and items between multiple domains to transfer their cluster-level preferences as auxiliary sources of information so that the cross-domain latent factors can be learned.

- Heterogeneous Cross-domain Rating Prediction. We also conduct the cross-domain rec- ommendation between domains with distinct characteristics, in addition to similar domains dis- cussed in the previous paragraph. We refer to this experiment as heterogeneous cross-domain rating prediction. That said, given a source domain C, we aim at predicting the ratings of new users in a target domain D whose rating preferences and item categories are much different from source domainC

- Third, in our current work, we perform MF at the online stage to obtain the new user’s latent factor pXT, rather than derive the latent factor of new users from the old latent matrix. In fact, we can derive the new latent factor of the new user via the technique of incremental matrix factor- ization [31]. The incremental matrix factorization can allow us to derive the latent factor of new user without performing MF again at the online stage. In the practical application, the incremental matrix factorization will be suggested.

- First, PLSR belongs to prediction by linear regression. For those user-rating data possessing the non-linearity property, the proposed PLSR-Latent may not lead to satisfying performance. We plan to apply a kernel PLSR [26] to allow prediction with non-linear regression. Second, an limitation of PLSR-based methods is needing to first complete the rating matrices. That said, it is necessary to contain no missing values in the input matrices of PLSR.
- 