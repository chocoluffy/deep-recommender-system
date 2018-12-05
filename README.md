## nn regression prediction

简单来说，regression用error = ‘MSE’； 
multi-label,用 error=‘binary_cross_entropy’

*各类regression的prediction方法*

* [ElasticNet (LB 0.547+) and feature importance | Kaggle](https://www.kaggle.com/den3b81/elasticnet-lb-0-547-and-feature-importance)  Elastic Net以及XGBoost可以用来找到哪些feature是最合适的。

* [stacked then averaged models ~ 0.5697 | Kaggle](https://www.kaggle.com/tobikaggle/stacked-then-averaged-models-0-5697)

* [using XGBOOST FOR  regression | Kaggle](https://www.kaggle.com/fashionlee/using-xgboost-for-regression) 很干净的一个使用XGBoost过程的一个展示。
很多时候csv的处理都是很类似的：
	* 利用pandas来read然后drop一些field来得到最后的关键features. 
	* 然后pop得到label。
	* 可以用得到的final data来进行train\test split。
	* 然后就是更换classifier。 

* [How Autoencoders Work: Intro and UseCases | Kaggle](https://www.kaggle.com/shivamb/how-autoencoders-work-intro-and-usecases) 一个关于auto-encoder应用方向的很好的介绍。对于image feature，我们经常用CNN。 而对于sequential data(like： time-series data or text data)， 用LSTM。
（LSTM的主要作用：Unlike other recurrent neural networks, the network’s internal gates allow the model to be trained successfully using backpropagation through time, or BPTT, and avoid the vanishing gradients problem.）