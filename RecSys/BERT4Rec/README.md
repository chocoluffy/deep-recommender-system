# BERT4Rec Sequential Recommendation with Bidirectional Encoder Representations from Transformer

评分：4/5。  
简介：将Bert双向Transformer的结构带入了推荐系统，并改变了目标用Cloze task来防止信息泄漏并可用于预测随机masked的item。  

- 其中一个关键的假设：用户行为序列并没有NLP那样严格的逻辑依赖关系，更多在视频推荐场景里强调的是相关性，发现性和多样性。在Transformer里依旧是单向的次序结构，只不过相比RNN来说，Transformer能够在每一步都连接过去所有的input，而不必将信息condense到一个hidden state里（信息噪音和流失）。
- Transformer的引入相比RNN对并行计算友好。复杂度从O(nd^2)变为O(n^2d)，其中n为序列长度而d为特征长度，对于短序列高维的特征表示来说，self-attention是一个选择。同时由于只引入了matrix multiplication的操作，对SGD运算友好。
- self attention本身其实和位置无关，只是一种扩大receptive field的方式，类似CNN中利用叠加起来的层次来一部部增大感受域，如果需要次序的概念，需要加入position encoding。
- 在推荐系统里，position embedding对应的意义是此操作距离用户当前时刻的时间差。这个position embedding在不同的场景应当有不同的意义和改进技巧。比如在图片领域强调的是translation invariance那么相比absolute position更合适的其实是relative position encoding等等。



# highlights

Although these methods achieve satisfactory results, they often assume a rigidly ordered sequence which is not always practical

jointly conditioning on both left and right context in deep bidirectional model would make the training become trivial since each item can indirectly “see the target item”. To address this problem, we train the bidirectional model using the Cloze task, predicting the masked items in the sequence jointly conditioning on their left and right context.

Early works usually adopt Markov chains (MCs) to model users’behavior sequences for predicting their next behaviors [11, 4.  Such methods usually make a strong simplified assumption, combining previous items independently which often hurts the recommendation accuracy

The choices of items in a user’s historical interactions may not follow a rigid order assumption [1

To address this problem, we introduce the Cloze task to take the place of the objective in unidirectional models (i.e., sequentially predicting the next item). Specifically, we randomly mask some items in the input sequences, and then predict only those masked items based on their surrounding context. Wilson L. Taylor. 1953. âĂĲCloze ProcedureâĂİ: A New Tool for Measuring Readability. Journalism Bulletin 30, 4 (1953), 415–433.

to introduce deep bidirectional sequential model and Cloze objective into the field of recommendation systems.

Among various CF methods, Matrix Factorization (MF) is the most popular one, which projects users and items into a shared vector space and estimate a user’s preference on an item by the inner product between their vectors [

The early pioneer work is a two-layer Restricted Boltzmann Machines (RBM) for collaborative filtering, proposed by Salakhutdinov et al. in Netflix Prize2. Ruslan Salakhutdinov, Andriy Mnih, and Geoffrey Hinton. 2007. Restricted Boltzmann Machines for Collaborative Filtering. In Proceedings of ICML. 791–798.

Position-wise Feed-Forward Network. As described above, the self-attention sub-layer is mainly based on linear projections.
stackoverflow.com/questions/46121283/what-is-linear-projection-in-convolutional-neural-network 

L.	Specifically, multi-head attention first linearly projects Hlinto h subspaces, with different, learnable linear projections,

our BERT4Rec uses bidirectional self-attention to jointly combine both left and right context. In this way, our proposed model can alleviate the limitation of previous methods, a rigid order assumption.

self-attention mechanism endows BERT4Rec with the capability to directly capture the dependencies in any distances. This mechanism results in a global receptive field, while CNN based methods like Caser usually have a limited receptive field. In addition, in contrast to RNN based methods, self-attention is straightforward to parallelize.

BERT4Rec is stacked by L bidirectional Transformer layers. At each layer, it iteratively revises the representation of every position by exchanging information across all positions at the previous layer in parallel with the Transformer layer.

Given the interaction history Su, sequential recommendation aims to predict the item that user u will interact with at time step nu+ 1. It can be formalized as modeling the probability: p?v(u) nu+1= v| Su ? over all possible items for user u at time step nu+ 1.

SASRec is closely related to our work. However, it is still a unidirectional model using a casual attention mask. While our proposed model learns to encode users’ historical records from both directions with the help of Cloze task.

In contrast, Transformer and BERT model the text sequence relying entirely on multi-head self-attention and achieve state-of-the-art results on tasks like machine translation and sentence classification.

Unfortunately, none of above methods is for sequential recommendation since they all ignore the order in users’ behaviors.

we apply a Position-wise Feed-Forward Network to the outputs of the self-attention sub-layer, separately and identically at each position. It consists of two affine transformations with a Gaussian Error Linear Unit (GELU) activation in between

In order to train a deep bidirectional sequential model, we introduce a new objective: Cloze task (also known as “Masked Language Model” in). It is a test consisting of a portion of language with some words removed

, jointly conditioning on both left and right context in a bidirectional model would cause the final output representation of each item contain the information of the target item.

Unlike traditional sequential recommendation model predicting the next t + 1 item given the first t items, we predict the masked items vtbase on hL tas shown in Figure 2b.

In this work, we use the learnable positional embeddings instead of the fixed sinusoid embeddings in for better performances.

An additional advantage for Cloze task is that it can generate more samples to train the model. Assuming a sequence of length n, conventional sequential predictions in Figure 2c and 2d produce n unique samples for training, while our BERT4Rec can obtain?n k ? samples (if we randomly mask k items). This allows us to learn a more powerful bidirectional representation model.

Specifically, given an interaction sequence Su= [v(u) 1, . . . ,v( for user u, we use item vu nufor test, vu nu−1for hyper-parameter tuning, and [v(u) 1, . . . ,v( for training.

To make the sampling reliable and representative, these 100 negative items are sampled according to their popularity.

Evaluation Metrics. To evaluate the ranking list of all the models, we employ a variety of evaluation metrics, including Hit Ratio (HR), Normalized Discounted Cumulative Gain (NDCG), and Mean Reciprocal Rank (MRR).

Here comes a question: do the gains come from the bidirectional self-attention model or from the Cloze objective?

This demonstrates the importance of bidirectional representations for sequential recommendation.

A larger hidden dimensionality does not necessarily lead to better model performance, especially on sparse datasets like Beauty and Steam. This is probably caused by overfitting.

Beauty prefers a smaller N = 20, while ML-1m achieves the best performances on N = 200. This indicates that user’s behavior is affected by more recent items on short sequence datasets, and less recent items for long sequence datasets.

A valuable direction is to incorporate rich item features (e.g., category and price for products, cast for movies) into BERT4Rec instead of just modeling item ids. Another interesting direction for the future work would be introducing user component into the model for explicit user modeling when the users have multiple sessions.

Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and Noam Shazeer. 2018. Generating Wikipedia by Summarizing Long Sequences. In Proceedings of ICLR.

Qiao Liu, Yifu Zeng, Refuoe Mokhosi, and Haibin Zhang. 2018. STAMP: ShortTerm Attention/Memory Priority Model for Session-based Recommendation. In Proceedings of KDD. ACM, New York, NY, USA, 

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.  CoRR abs/1810.04805 (2018).
 Wang-Cheng Kang and Julian McAuley. [. Self-Attentive Sequential Recommendation. In Proceedings of ICDM. 197–206.