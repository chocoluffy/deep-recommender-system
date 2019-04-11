# Collaborative Deep Learning for Recommender Systems

评分：4/5。  
简介：针对rating和content information matrix，设计MAP(Maximum A Priori)的objective function来改善user embedding。相比传统collaborative filtering不擅长直接处理稀疏rating输入，CDL通过更好地结合content information可以得到更好的rating prediction。  

- 引入SDAE(Stacked Denoising Auto Encoder)来获得item的compact feature。随机初始化单位user embedding，长度等同于SDAE中encoder的输出维度，使得`uTv = r`为目标来训练。其中v为encoder输出，及item的compact feature。构造MAP的objective利用EM交替更新参数。
- 参数初始化都用bayesian model。整体上是hierachial bayesian model。
- 使用recall作为evaluation metric，即在推荐结果中relevant的数量占全部relevant item的数量。对比precision。

# highlights

- Conventional CF-based methods use the ratings given to items by users as the sole source of information for learning to make recommendation. However, the ratings are often very sparse in many applications, causing CF-based methods to degrade significantly in their recommendation performance. To address this sparsity problem, auxiliary information such as item content information may be utilized.

- we generalize recent advances in deep learning from i.i.d. input to non-i.i.d. (CF-based) input and propose in this paper a hierarchical Bayesian model called collaborative deep learning (CDL), which jointly performs deep representation learning for the content information and collaborative filtering for the ratings (feedback) matrix.

- Existing methods for RS can roughly be categorized into three classes [6]: content-based methods, collaborative filtering (CF) based methods, and hybrid methods.

- Content-based methods [17] make use of user profiles or product descriptions for recommendation. CF-based methods [23, 27] use the past activities or preferences, such as user ratings on items, without using user or product content information. Hybrid methods [1, 18, 12] seek to get the best of both worlds by combining content-based and CF-based methods.

- CF-based methods do have their limitations. The prediction accuracy often drops significantly when the ratings are very sparse. Moreover, they cannot be used for recommending new products which have yet to receive rating information from users. Consequently, it is inevitable for CF-based methods to exploit auxiliary information and hence hybrid methods have gained popularity in recent years.

- we may further divide hybrid methods into two sub-categories: loosely coupled and tightly coupled methods.

- Although they are more appealing than shallow models in that the features can be learned automatically (e.g., effective feature representation is learned from text content), they are inferior to shallow models such as CF in capturing and learning the similarity and implicit relationship between items.

- We first present a Bayesian formulation of a deep learning model called stacked denoising autoencoder (SDAE) [32].

With this, we then present our CDL model which tightly couples deep representation learning for the content information and collaborative filtering for the ratings (feedback) matrix, allowing two-way interaction between the two.

- CDL can simultaneously extract an effective deep feature representation from content and capture the similarity and implicit relationship between items (and users).

- Given part of the ratings in R and the content information Xc, the problem is to predict the other ratings in R.

- of the posterior probability is equivalent to minimization of the reconstruction error with weight decay taken into consideration.