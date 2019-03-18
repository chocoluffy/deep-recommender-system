# 目录

## Kaggle
- [Kaggle Jupyter技巧总结：经典机器学习模型，Pipeline, GridSearch, Ensemble等](https://github.com/chocoluffy/kaggle-notes/tree/master/Kaggle)

## Deep Learning
- [Softmax的numpy实现, 以及SGD、minibatch](https://github.com/chocoluffy/kaggle-notes/blob/master/DL/Softmax.md)
- [Ridge Regression的实现](https://github.com/chocoluffy/kaggle-notes/blob/master/DL/RidgeRegression.md)

## RecSys
- [Practical Lessons from Predicting Clicks on Ads at Facebook](https://github.com/chocoluffy/kaggle-notes/tree/master/RecSys/predicting-clicks-facebook)

评分：5/5。
简介：Facebook提出的CTR预估模型，GBDT + Logistic Regression。

- 加深了对entropy的理解，以及在CTR领域使用normalized entropy的实践。
- 学习到了GBM和LR的结合。用boosted decision tree来主要负责supervised feature learning有很大的优势。之后对接的LR + SGD可以作为online learning保持日常更新训练保证data freshness。
- 对引入prior的理解。[1] integrating knowledge. [2] avoid overfitting, similar to regularization.
- 在online learning领域，对比LR + SGD和BOPR（Bayesian online learning scheme for probit regression）的优劣势。

# 总结

## classification

for supervised learning classification (such as the current problem) include techniques such as:

- logistic regression and penalized logistic regression
- linear discriminant analysis
- decision trees (CART, CHAID, C5.0)
- random forests
- gradient boosted machines
- support vector machines
- neural networks (and deep learning)

## regression (continuous outcome)

for supervised learning for continuous outcome problems (e.g., the second problem) include techniques such as:

- linear regression, ridge regression, lasso, elastic net
- partial least squares regression(PLSR), principal component regression(PCR)
- decision trees (CART, CHAID, C5.0)
- multiple adaptive regression splines
- random forests
- gradient boosted machines
- support vector machine regression
- neural networks (and deep learning)
