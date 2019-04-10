# Wide & Deep Learning for Recommender Systems

评分：5/5。  
简介：利用logistic regression针对广度的交叉特征(cross product transformation)，利用NN负责深度特征挖掘，并同时进行joint training。  

- 提出Wide & Deep的解决方案来改善memorization(relevancy)和generality(diversity)的表现。改善传统embedding容易因为稀疏输入而over-generalize的问题。
- 介绍了工业上推荐系统的流程，先通过retrival从数据库选出初步的candidate，O(100)的量级；然后再通过rank的模型将candidate进行精排返回前十作为结果，Wide & Deep是在这个rank阶段的一种方案。
- 每一次retrain时利用上一次的weight初始化，以减少训练时间，类比transfer learning。
- 为了线上的低延迟在10ms量级，采取多线程small batch的方式跑inference，而不是将所有candidate放在同一个batch里跑。最终batch size为50，4个线程可以达到14ms的表现。
- Matrix Factorization里通过dot product引入interaction的特征，其根本目的是为了引入non-linearity。但这部分可以用NN更好的完成。

# highlights

- Memorization of fea- ture interactions through a wide set of cross-product feature transformations are effective and interpretable, while gener- alization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize bet- ter to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features.

- In this paper, we present Wide & Deep learning—jointly trained wide linear models and deep neural networks—to combine the benefits of mem- orization and generalization for recommender systems.

- A recommender system can be viewed as a search ranking system, where the input query is a set of user and contextual information, and the output is a ranked list of items.

- a query, the recommendation task is to find the relevant items in a database and then rank the items based on certain objectives, such as clicks or purchases.

One challenge in recommender systems, similar to the gen- eral search ranking problem, is to achieve both memorization and generalization. Memorization can be loosely defined as learning the frequent co-occurrence of items or features and exploiting the correlation available in the historical data.

Generalization, on the other hand, is based on transitivity of correlation and explores new feature combinations that ∗Corresponding author: hengtze@google.com have never or rarely occurred in the past. Recommenda- tions based on memorization are usually more topical and directly relevant to the items on which users have already performed actions. Compared with memorization, general- ization tends to improve the diversity of the recommended items.

- generalized linear models such as logistic regression are widely used because they are sim- ple, scalable and interpretable. The models are often trained on binarized sparse features with one-hot encoding. E.g., the binary feature “user_installed_app=netflix” has value 1 if the user installed Netflix. Memorization can be achieved effectively using cross-product transformations over sparse features, such as AND(user_installed_app=netflix, impres- sion_app=pandora”)

- but manual feature engineer- ing is often required. One limitation of cross-product trans- formations is that they do not generalize to query-item fea- ture pairs that have not appeared in the training data.

- Embedding-based models, such as factorization machines [5] or deep neural networks, can generalize to previously un- seen query-item feature pairs by learning a low-dimensional dense embedding vector for each query and item feature, with less burden of feature engineering. However, it is dif- ficult to learn effective low-dimensional representations for queries and items when the underlying query-item matrix is sparse and high-rank, such as users with specific preferences or niche items with a narrow appeal. In such cases, there should be no interactions between most query-item pairs, but dense embeddings will lead to nonzero predictions for all query-item pairs, and thus can over-generalize and make less relevant recommendations

- Since there are over a million apps in the database, it is intractable to exhaustively score every app for every query within the serving latency requirements (often O(10) mil- liseconds). Therefore, the first step upon receiving a query is retrieval. The retrieval system returns a short list of items that best match the query using various signals, usually a combination of machine-learned models and human-defined rules. After reducing the candidate pool, the ranking sys- tem ranks all items by their scores. The scores are usually P (y|x), the probability of a user action label y given the features x, including user features (e.g., country, language, demographics), contextual features (e.g., device, hour of the day, day of the week), and impression features (e.g., app age, historical statistics of an app). In this paper, we focus on the ranking model using the Wide & Deep learning framework.

- 

- Note that there is a distinction be- tween joint training and ensemble. In an ensemble, indi- vidual models are trained separately without knowing each other, and their predictions are combined only at inference time but not at training time. In contrast, joint training optimizes all parameters simultaneously by taking both the wide and deep part as well as the weights of their sum into account at training time. There are implications on model size too: For an ensemble, since the training is disjoint, each individual model size usually needs to be larger (e.g., with more features and transformations) to achieve reasonable accuracy for an ensemble to work. In comparison, for joint training the wide part only needs to complement the weak- nesses of the deep part with a small number of cross-product feature transformations, rather than a full-size wide model.

- Vocabularies, which are tables mapping categorical fea- ture strings to integer IDs, are also generated in this stage.

The system computes the ID space for all the string features that occurred more than a minimum number of times. Con- tinuous real-valued features are normalized to [0, 1] by map- ping a feature value x to its cumulative distribution function P (X ≤ x), divided into nq quantiles.

- To tackle this challenge, we implemented a warm-starting system which initializes a new model with the embeddings and the linear model weights from the previous model.

- In order to serve each request on the order of 10 ms, we optimized the performance using multithreading parallelism by running smaller batches in parallel, instead of scoring all candidate apps in a single batch inference step.

- The idea of combining wide linear models with cross- product feature transformations and deep neural networks with dense embeddings is inspired by previous work, such as factorization machines [5] which add generalization to linear models by factorizing the interactions between two variables as a dot product between two low-dimensional embedding vectors. In this paper, we expanded the model capacity by learning highly nonlinear interactions between embeddings via neural networks instead of dot products.

- T. Mikolov, A. Deoras, D. Povey, L. Burget, and J. H.

Cernocky. Strategies for training large scale neural network language models. In IEEE Automatic Speech Recognition & Understanding Workshop, 2011.

- In language models, joint training of recurrent neural net- works (RNNs) and maximum entropy models with n-gram features has been proposed to significantly reduce the RNN complexity (e.g., hidden layer sizes) by learning direct weights between inputs and outputs [4]. In computer vision, deep residual learning [2] has been used to reduce the difficulty of training deeper models and improve accuracy with shortcut connections which skip one or more layers. Joint training of neural networks with graphical models has also been applied to human pose estimation from images [6

- H. Wang, N. Wang, and D.-Y. Yeung. Collaborative deep learning for recommender systems. In Proc. KDD, pages 1235–1244, 2015
