# Behavior Sequence Transformer for E-commerce Recommendation in Alibaba

评分：3/5。
简介：将Transformer的self attention结构应用在推荐系统典型的Wide & Deep网络结构中。

- 结合了position embedding，用距离当前推荐时间的时间差作为位置信息。
- 采用的是内部的attention机制，也即Q = K = V = embedding，其中dot product计算的是物品之间的相似程度。最后采用multi-head的做法，类比CNN中使用的多个kernel得到多个feature map，multi-head使得能够探索出embedding不同位置的特性。注意的一点是，attention同样可以引入外部的embedding，只要保证key和value是一一对应的即可，可以利用外部embedding来升、降维度。
- 最终的目标是预测目标产品的CTR的概率，适合电商的环境。对于视频推荐的场景，可以尝试youtube那篇文章的目标，即expected watch time。

# highlights

[2] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al.

2016. Wide & deep learning for recommender systems. , 7–10 pages.

[3] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In RecSys. 191–198.

[5] Mihajlo Grbovic and Haibin Cheng. 2018. Real-time personalization using em- beddings for search ranking at Airbnb. In KDD. 311–320.

[15] Jizhe Wang, Pipei Huang, Huan Zhao, Zhibo Zhang, Binqiang Zhao, and Dik Lun Lee. 2018. Billion-scale commodity embedding for e-commerce recommendation in alibaba. In KDD. 839–848.

In the era of deep learning, embedding and MLP have been the standard paradigm for industrial RSs: large numbers of raw features are embedded into low-dimensional spaces as vectors, and then fed into fully connected layers, known as multi layer perceptron (MLP), to predict whether a user will click an item or not. The representative works are wide and deep learning (WDL) networks [2] from Google and Deep Interest networks (DIN) from Alibaba [17]

As introduced in [15], the RSs in Alibaba are a two-stage pipeline: match and rank.

In WDL [2], they simply concatenates all features without capturing the order information among users’behavior sequences. In DIN [17], they proposed to use attention mechanism to capture the similarities between the candidate item and the previously clicked items of a user, while it did not consider the sequential nature underlying the user’s behavior sequences.

[17] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In KDD. 1059–1068.

we apply the self-attention mechanism to learn a better representation for each item in a user’s behavior sequence by considering the sequential information in embedding stage, and then feed them into MLPs to predict users’ re- sponses to candidate items.

In the rank stage, we model the recommendation task as Click- Through Rate (CTR) prediction problem, which can be defined as follows: given a user’s behavior sequence S(u) = {v1,v2, ...,vn} clicked by a user u, we need to learn a function, F , to predict the probability of u clicking the target item vt, i.e., the candidate one. Other Features include user profile, context, item, and cross features.

The key difference between BST and WDL is that we add transformer layer to learn better representa- tions for users’ clicked items by capturing the underlying sequential signals.

To predict whether a user will click the target itemvt, we model it as a binary classification problem, thus we use the sigmoid function

we use the multi-head attention: S =MH(E) = Concat(head1,head2, · · · ,headh)WH, (2) headi=Attention(EWQ, EWK, EWV), (3) where the projection matrices WQ, WK, WV∈ Rd×d, and E is the embedding matrices of all items, and h is the number of heads.

Note that the position value of item viis computed as pos(vi) = t(vt) − t(vi), 2 where t(vt) represents the recommending time and t(vi) the times- tamp when user click item vi.

like the user profile features, item features, context features, and the combination of different features, i.e., the cross features1

DIN tried to capture different similarities between the previously clicked items and the target item.

Since the proposal of WDL [2], a se- ries of works have been proposed to improve the CTR with deep learning based methods, e.g., DeepFM [6], XDeepFM [9], Deep and Cross networks [16], etc. However, all these previous works focus on feature combinations or different architectures of neural net- works, ignoring the sequential nature of users’ behavior sequence in real-world recommendation scenarios.
