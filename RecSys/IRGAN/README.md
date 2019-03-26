# IRGAN - A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models

评分：5/5。
简介：将GAN应用在information retrieval上。SIGIR2017满分论文。

- 巧妙地构造了一个minmax的机制。discriminator负责判断一个document是否well-matched，通过maximum likelihood。而对generator来说，则是更新参数minimize这个maximum likelihood。可以借鉴的点，在于如何设计的likelihood的期望。Discriminator其实核心就是一个binary classifier，然后利用logistic转换到(0, 1)的值域范围，就可以设计`log(D(d|q)) + log(1-D(d'|q))`的likelihood来达到目标！（其中d'为generator的样本，d为ground truth distribution的样本）。理解的思路其实很简单，generator生成的d'是试图欺骗discriminator的，因此如果D判定d'为well-matched，则因此可以引入large loss来penalize discriminator，也是`log(1-D(d'|q))`的设计思路。
- 相比传统的GAN在continuous latent space生成图片，IRGAN的generator则是在document pool选择最相关的relevant document，是离散的。于是引入RI里的policy graident来descent。
- 利用hierachial softmax来降低softmax的复杂度。传统的softmax和所有的document相关，而hierachial softmax可以降至log(D).
- MLE(Maximum Likelihood Estimation)是MAP(Maximum A Priori)的一种特殊情况，即Prior为uniform distribution的。MAP既考虑了likelihood还考虑了参数的Prior。

# highlights

- On one hand, the discriminative model, aiming to mine signals from labelled and unlabelled data, provides guidance to train the generative model towards fting the underlying relevance distribution over documents given the query.  On the other hand, the generative model, acting as an atacker to the current discriminative model, generates difcult examples for the discriminative model in an adversarial way by minimising its discrimination objective  

- Our experimental results have demonstrated signifcant performance gains as much as 23.96% on Precision@5 and 15.50% on MAP over strong baselines in a variety of applications including web search, item recommendation, and question answering.  

- token is independently generated to form a relevant document [35]. Statistical language ∗Te corresponding authors: J. Wang and W. Zhang.  models of text retrieval consider a reverse generative process from a document to a query: d → q, typically generating query terms from a document (i.e., the query likelihood function) [32, 48]. In the related work of word embedding, word tokens are generated from their context words [28]. In the application of recommender systems, we also see that a recommended target item (in the original document identifer space) can be generated/selected from known context items [2]  

- Te modern school of thinking in IR recognises the strength of machine learning and shifs to a discriminative (classifcation) solution learned from labelled relevant judgements or their proxies such as clicks or ratings. It considers documents and queries jointly as features and predicts their relevancy or rank order labels from a large amount of training data: q +d → r, where r denotes relevance and symbol + denotes the combining of features.  

- Tree major paradigms of learning to rank are pointwise, pairwise, and listwise.  

- Te classic school of thinking is to assume that there is an underlying stochastic generative process between documents and information needs (clued by a query)  

- On the other hand, the generative retrieval model pθ(d|q,r) acts as a challenger who constantly pushes the discriminator to its limit. Iteratively it provides the most difcult cases for the discriminator to retrain itself by adversarially minimising the objective function.  

- Note that our minimax game based approach is fundamentally diferent from the existing game-theoretic IR methods [26, 47], in the sense that the existing approaches generally try to model the interaction between user and system, whereas our approach aims to unify generative and discriminative IR models.  

- fϕ(q,d), which, in contrary, tries to discriminate well-matched query-document tuples (q,d) from ill-matched ones,  

- the generative retrieval model would try to generate (or select) relevant documents that look like the groundtruth relevant documents and therefore could fool the discriminative retrieval model, whereas the discriminative retrieval model would try to draw a clear distinction between the ground-truth relevant documents and the generated ones made by its opponent generative retrieval model.  

- Te underlying true relevance distribution can be expressed as conditional probability ptrue(d|q,r), which depicts the (user’s) relevance preference distribution over the candidate documents with respect to her submited query. Given a set of samples from ptrue(d|q,r) observed as the training data, we can try to construct two types of IR models:  

- pθ(d|q,r), which tries to generate (or select) relevant documents, from the candidate pool for the given query q, as specifed later in Eq. (8); in other words, its goal is to approximate the true relevance distribution over documents ptrue(d|q,r) as much as possible.  

- It is worth mentioning that unlike GAN [13, 18], we design the generative model to directly generate known documents (in the 2 document identifer space) not their features, because our work here intends to select relevant documents from a given document pool  

- As the sampling of d is discrete, it cannot be directly optimised by gradient descent as in the original GAN formulation. A common approach is to use policy gradient based reinforcement learning (REINFORCE) [42, 44]. Its gradient is derived as follows  

- In each epoch of training, the generator tries to generate samples close to the discriminator’s decision boundary to confuse its training next round, while the discriminator tries to score down the generated samples  

- Such a complexity can largely be reduced to O(NK log M) by applying hierarchical sofmax [28] in the sampling process of the generator.  

- Our work is diferent in the following three aspects.  First, the generative retrieval process is stochastic sampling over discrete data, i.e., the candidate documents, which is diferent from the deterministic generation based on the sampled noise signal in the original GAN  

- the unifed training scheme of two schools of IR models ofers the potential of geting beter retrieval models, because (i) the generative retrieval adaptively provides diferent negative samples to the discriminative retrieval training, which is strategically diverse compared with static negative sampling [3, 34] or dynamic negative sampling using the discriminative retrieval model itself [4, 50, 51]  

- NCE is usually leveraged as an efcient approximation to MLE when the later is inefcient, for example when the p.d.f is built by large-scale sofmax modelling.  

- while in IRGAN the generator-picked documents are regarded as negative samples to train the ranker;  

- In the web search scenario, each query-document pair (q,d) can be represented by a vector xq,d∈ Rk, where each dimension represents some statistical value of the query-document pair or either part of it, such as BM25, PageRank, TFIDF, language model score etc. We follow the work of RankNet [3] to implement a two-layer neural network for the score function:  


