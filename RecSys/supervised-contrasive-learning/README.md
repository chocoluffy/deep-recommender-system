# Supervised Contrastive Learning

评分：4/5。 
简介：Google家在SimCLR自监督对比学习（contrastive learning）的loss结构基础上，延伸到有监督学习中，并在多项图像任务中排列第一。论文本身和推荐系统无关，但是如果仔细推敲，公式非常接近bayesian personalized ranking loss（BPR loss），而其实这一类triplet loss正是contrasive learning的其中一个子应用，也即当只有一个正负样本对的时候，当正负样本对的数量增大到多个时，将sigmoid改进为softmax，即可推出该论文中自监督学习的公式了。

- triplet loss是contrasive learning的其中一个应用。即当只有一个正样本和一个负样本的时候。对当前mini-batch内的样本做对比学习。
- 相比于无监督的模式（SimCLR）的做法。
	- 需要一个encoder network，将image raw features映射为一个向量。比如图像中使用的是resnet，输出2048维度的向量。叠加一个norm layer，实验表明有助于提高指标。
	- 一个projection network，**将2048维度映射到128维度，后续接一个norm layer，因为这么做能够是的后续的内积操作 = cos可以等效于优化向量方向来训练。分子 -> 1也就是向量方向一致。**最终infer的时候会抛弃这个projection network，理由：**实验表明，使用encoder network的结果对后续下游任务更友好。**
- 启发：这个方式不局限于z = user embedding点乘item embedding。也适用于i2i的训练，其中z = 两个item embedding的相似度。
- 为什么这个有效果？因为主要来自于分母的设计，**需要最大化分子的同时，最小化分母；也就是分母中所有负样本中最难的那个样本会成为优化的上界。从而模型会去优化更难的样本。而不会说是难的负样本和容易的负样本贡献的梯度是一致的。**
- 在无监督的公式里面，分子是1个正样本:
![self-supervised-equations](https://raw.githubusercontent.com/chocoluffy/deep-recommender-system/master/RecSys/supervised-contrasive-learning/imgs/self-supervised-equations.png)
- 而在监督学习里面，把1个正样本拓展为这个mini batch里面的所有同属于这个class的正样本:
![supervised-equations](https://raw.githubusercontent.com/chocoluffy/deep-recommender-system/master/RecSys/supervised-contrasive-learning/imgs/supervised-equations.png)
- **公式非常接近bayesian personalized ranking loss。其中的区别是，bpr loss针对的是一对正负样本对，属于triplet loss。**
- **在binary classification的时候，sigmoid = softmax；也就是其实把BPR loss延展为多分类问题的时候，就改为softmax，也就是上图无监督学习中的公式了。也就是从反向角度去证明了，其实SimCLR的思路有效，其实等效于把一个二分类的优化问题，转换为了多分类优化**，其中其他的分类是包含了较难的负样本和较容易的负样本。
![bpr-loss](https://raw.githubusercontent.com/chocoluffy/deep-recommender-system/master/RecSys/supervised-contrasive-learning/imgs/bpr-loss.png)
