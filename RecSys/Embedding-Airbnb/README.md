# Real-time Personalization using Embeddings for Search Ranking at Airbnb

评分：5/5。  
简介：对listing做embedding，将user type，listing type以及query在同一个vector space构建embedding，以实时更新搜索结果并提高准度。KDD 2018 best paper。

- 利用in-session signal，将用户行为（包括点击、最终达成交易以及被拒绝）模拟为一个时序序列，类比word2vec中的单个句子。然后利用skip-gram和negative sampling来进行word2vec模型的训练。
- 对实际场景的精准观察是很多机制设计的灵感来源。比如：
    - 将最终的bookings作为global context添加进每一个sliding window的训练。
    - 因为旅游目的地的搜索是congregated search，于是添加在target area区域内的negative sampling sets。
    - 将奖赏(vetor距离更新)和惩罚(vector相互远离)加入进objective function的设计。
    - 考虑到用户booking的稀疏性，采用user type(users' many-to-one relation)，并将listing type和query type并入同一序列进行训练，使得embedding在同一个vector space，以允许用户在搜索时可以提供语义上最接近的结果(而不是简单匹配)，并改善最相近listing carousel模块的推荐结果。

# highlights

- When learning listing embeddings we treat the booked listing as global context that is always being predicted as the window moves over the session.

- More recently, the concept of embeddings has been extended beyond word representations to other applications outside of NLP domain. Researchers from the Web Search, E-commerce and Mar- ketplace domains have quickly realized that just like one can train word embeddings by treating a sequence of words in a sentence as context, same can be done for training embeddings of user ac- tions, e.g. items that were clicked or purchased [11, 18], queries and ads that were clicked [8, 9], by treating sequence of user actions as context.

- Finally, similar ex- tensions of embedding approaches have been proposed for Social Network analysis, where random walks on graphs can be used to learn embeddings of nodes in graph structure [12, 20].

- the objective of the model is to learn listing representations using the skip-gram model [17] by maximizing the objective function L over the entire set S of search sessions

- Time required to compute gradient ∇L of the objective function in (1) is proportional to the vocabulary size |V|, which for large vocabularies, e.g. several millions listing ids, is an infeasible task. As an alternative we used negative sampling approach proposed in [17], which significantly reduces computational complexity

- Users of online travel booking sites typically search only within a single market, i.e. location they want to stay at. As a consequence, there is a high probability that Dpcontains listings from the same market

- • First, booking sessions data Sbis much smaller than click sessions data S because bookings are less frequent events.

• Second, many users booked only a single listing in the past and we cannot learn from a session of length 1.

- sb= (utype1ltype1, . . . ,utypeMltypeM) ∈ Sbis defined as a se- quence of booking events, i.e. (user_type, listinд_type) tuples or- dered in time. Note that each session consists of bookings by same user_id, however for a single user_id their user_types can change over time, similarly to how listinд_types for the same listing can change over time as they receive more bookings.

- Dimensionality of listing embeddings was set to d = 32, as we found that to be a good trade-off between offline performance and memory needed to store vectors in RAM memory of search machines for purposes of real-time similarity calculations. Context window size was set to m = 5, and we performed 10 iterations over the training data.

- . More specifically, let us assume we are given the most recently clicked listing and listing candidates that need to be ranked, which contain the listing that user eventually booked. By calculating cosine similarities between embeddings of clicked listing and candidate listings we can rank the candidates and observe the rank position of the booked listing.

- similar listings were produced by finding the k-nearest neighbors in listing embedding space. Given learned listing embeddings, similar listings for a given listing l were found by calculating cosine similar- ity between its vector vland vectors vjof all listings from the same market that are available for the same set of dates (if check-in and check-out dates are set). The K listings with the highest similarity were retrieved as similar listings. The calculations were performed online and happen in parallel using our sharded architecture, where parts of embeddings are stored on each of the search machines.

- our Search Ranking Model, let us assume we are given training data about each search Ds = (xi,yi),i = 1...K, where K is the number of listings returned by search, xiis a vector containing features of the i-th listing result and yi ∈ {0, 0.01, 0.25, 1, −0.4} is the label assigned to the i-th listing result. To assign the label to a particular listing from the search result we wait for 1 week after search happened to observe the final outcome, which can be yi = 1 if listing was booked, yi = 0.25 if listing host was contacted by the guest but booking did not happen, y = −0.4 if listing host rejected the guest, yi= 0.01 is listing was clicked and yi= 0 if listing was just viewed but not clicked.

- Next, we formulate the problem as pairwise regression with search labels as utilities and use data D to train a Gradient Boosting Decision Trees (GBDT) model, using package4that was modified to support Lambda Rank. When evaluating different models offline, we use NDCG, a standard ranking metric, on hold-out set of search sessions, i.e. 80% of D for training and 20% for testing.

- Finally, we trained a new GBDT Search Ranking model with embedding features added. Feature importances for embedding features (ranking among 104 features) are shown in Table 7
