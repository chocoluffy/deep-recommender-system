# Practical Lessons from Predicting Clicks on Ads at Facebook

评分：4/5。

- 加深了对entropy的理解，以及在CTR领域使用normalized entropy的实践。
- 学习到了GBM和LR的结合。用boosted decision tree来主要负责supervised feature learning有很大的优势。之后对接的LR + SGD可以作为online learning保持日常更新训练保证data freshness。
- 对引入prior的理解。[1] integrating knowledge. [2] avoid overfitting, similar to regularization.
- 在online learning领域，对比LR + SGD和BOPR（Bayesian online learning scheme for probit regression）的优劣势。

# highlights

- a model which combines decision trees with logistic regression, 

- the most important thing is to have the right features: those capturing historical information about the user or ad dominate other types of features. 

- The 2007 seminal papers by Varian [11] and by Edelman et al. [4] describe the bid and pay per click auctions pioneered by Google and Yahoo! 

- As a consequence of this, the volume of ads that are eligible to be displayed when a user visits Facebook can be larger than for sponsored search. 
google: display ads by query;  fb: display ads by user’s demographic and interests; (different level of relevancy)

- we would first build a cascade of classifiers of increasing computational cost. In this paper we focus on the last stage click prediction model of a cascade classifier, that is the model that produces predictions for the final set of candidate ads. 

- In this work, we use Normalized Entropy (NE) and calibration as our major evaluation metric. 

- The background CTR is the average empirical CTR of the training data set. 
Click-through rate (CTR) is the ratio of users who click on a specific link to the number of total users who view a page, email, or advertisement. It is commonly used to measure the success of an online advertising campaign for a particular website as well as the effectiveness of email campaigns.

- decision trees are very powerful input feature transformations, that significantly increase the accuracy of probabilistic linear classifiers. 

- There are two simple ways to transform the input features of a linear classifier in order to improve its accuracy. For continuous features, a simple trick for learning non-linear transformations is to bin the feature and treat the bin index as a categorical feature. The linear classifier effectively learns a piece-wise constant non-linear map for the feature.  It is important to learn useful bin boundaries, and there are many information maximizing ways to do this.  The second simple but effective transformation consists in building tuple input features. For categorical features, the brute force approach consists in taking the Cartesian product, i.e. in creating a new categorical feature that takes as values all possible values of the original features. Not all combinations are useful, and those that are not can be pruned out. If the input features are continuous, one can do joint binning, using for example a k-d tree. 

- The boosted decision trees we use follow the Gradient Boosting Machine (GBM) 

- We can understand boosted decision tree based transformation as a supervised feature encoding that converts a real-valued vector into a compact binary-valued vector. A traversal from root node to a leaf node represents a rule on certain features. Fitting a linear classifier on the binary vector is essentially learning weights for the set of rules. Boosted decision trees are trained in a batch manner. 

- The boosted decision trees can be trained daily or every couple of days, but the linear classifier can be trained in near real-time by using some flavor of online learning. 

- The global learning rate fails mainly due to the imbalance of number of training instance on each features. 

- To achieve the stream-to-stream join the system utilizes a HashQueue consisting of a First-InFirst-Out queue as a buffer window and a hash map for fast random access to label impressions. 

- One advantages of LR over BOPR is that the model size is half, given that there is only a weight associated to each sparse feature value, rather than a mean and a variance. Depending on the implementation, the smaller model size may lead to better cache locality and thus faster cache lookup 

- One important advantage of BOPR over LR is that being a Bayesian formulation, it provides a full predictive distribution over the probability of click. This can be used to compute percentiles of the predictive distribution, which can be used for explore/exploit learning schemes [3]. 

- T. Graepel, J. Qui˜ nonero Candela, T. Borchert, and R. Herbrich. Web-scale bayesian click-through rate prediction for sponsored search advertising in Microsoft’s Bing search engine. In ICML, pages 13–20, 2010 

- Some example contextual features can be local time of day, day of week, etc. Historical features can be the cumulative number of clicks on an ad, etc. 

- historical features provide considerably more explanatory power than contextual features. 

- It should be noticed that contextual features are very important to handle the cold start problem. 

- A common technique used to control the cost of training is reducing the volume of training data.  In this section we evaluate two techniques for down sampling data, uniform subsampling and negative down sampling 

- Moreover, the data volume demonstrates diminishing return in terms of prediction accuracy. 

- In this part, we investigate the use of negative down sampling to solve the class imbalance problem. 

- Boosted decision trees give a convenient way of doing feature selection by means of feature importance. One can aggressively reduce the number of active features whilst only moderately hurting prediction accuracy. 

