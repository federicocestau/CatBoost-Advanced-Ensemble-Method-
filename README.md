# CatBoost-Advanced-Ensemble-Method-
CatBoost is the first Russian machine learning algorithm developed to be open source. The algorithm was developed in the year 2017 by machine learning researchers and engineers at Yandex (a technology company).
The term CatBoost is an acronym that stands for "Category” and “Boosting.” Does this mean the “Category’ in CatBoost means it only works for categorical features?
The answer is, “No.”
CatBoost has two main features, it works with categorical data (the Cat) and it uses gradient boosting (the Boost). Gradient boosting is a process in which many decision trees are constructed iteratively. Each subsequent tree improves the result of the previous tree, leading to better results. CatBoost improves on the original gradient boost method for a faster implementation.
According to the CatBoost documentation, CatBoost supports numerical, categorical, and text features but has a good handling technique for categorical data. 
The CatBoost algorithm has quite a number of parameters to tune the features in the processing stage.
"Boosting" in CatBoost refers to the gradient boosting machine learning. Gradient boosting is a machine learning technique for regression and classification problems. 
Which produces a prediction model in an ensemble of weak prediction models, typically decision trees. 
Here we would look at the various features the CatBoost algorithm offers and why it stands out:
Robust,
CatBoost can improve the performance of the model while reducing overfitting and the time spent on tuning.  
CatBoost has several parameters to tune. Still, it reduces the need for extensive hyper-parameter tuning because the default parameters produce a great result.
Overfitting is a common problem in gradient boosting, especially when the dataset is small or noisy. CatBoost has several features that help reduce overfitting.
One of them is a novel gradient-based regularization technique called ordered boosting, which penalizes complex models that overfit the data. Another feature is the use of per-iteration learning rate, which allows the model to adapt to the complexity of the problem at each iteration.
Automatic Handling of Missing Values,
Missing values are a common problem in real-world datasets. Traditional gradient boosting frameworks require imputing missing values before training the model. CatBoost, however, can handle missing values automatically. 
During training, it learns the optimal direction to move along the gradient for each missing value, based on the patterns in the data.
Accuracy,
The CatBoost algorithm is a high performance and greedy novel gradient boosting implementation. 
Categorical Features Support,
The key features of CatBoost is one of the significant reasons why it was selected by many boosting algorithms such as LightGBM,  XGBoost algorithm,etc.
With other machine learning algorithms. After preprocessing and cleaning your data, the data has to be converted into numerical features so that the machine can understand and make predictions.
This is same like, for any text related models we convert the text data into to numerical data it is know as word embedding techniques.
CatBoost overcomes a limitation of other decision tree-based methods in which, typically, the data must be pre-processed to convert categorical string variables to numerical values, one-hot-encodings, and so on. This method can directly consume a combination of categorical and non-categorical explanatory variables without preprocessing. It preprocesses as part of the algorithm. CatBoost uses a method called ordered encoding to encode categorical features. Ordered encoding considers the target statistics from all the rows prior to a data point to calculate a value to replace the categorical feature.

This process of encoding or conversion is time-consuming. CatBoost supports working with non-numeric factors, and this saves some time plus improves your training results.
Faster Training & Predictions,
Before the improvement of servers, the maximum number of GPUs per server is 8 GPUs. Some data sets are more extensive than that, but CatBoost uses distributed GPUs. 
This feature enables CatBoost to learn faster and make predictions 13-16 times faster than other algorithms.
Interpretability,
CatBoost provides some level of interpretability. It can output feature importance scores, which can help understand which features are most relevant for the prediction. 
It also supports visualization of decision trees, which can help understand the structure of the model.
Split,
The CatBoost algorithm introduced a unique system called Minimal Variance Sampling (MVS), which is a weighted sampling version of the widely used approach to regularization of boosting models, Stochastic Gradient Boosting.  
Also, Minimal Variance Sampling (MVS) is the new default option for subsampling in CatBoost.
With this technique, the number of examples needed for each iteration of boosting decreases, and the quality of the model improves significantly compared to the other gradient boosting models. 
The features for each boosting tree are sampled in a way that maximizes the accuracy of split scoring.
Leaf Growth,
Another unique characteristic of CatBoost is that it uses symmetric trees. This means that at every depth level, all the decision nodes use the same split condition.
The CatBoost algorithm grows a balanced tree. In the tree structure, the feature-split pair is performed to choose a leaf. 
The split with the smallest penalty is selected for all the level's nodes according to the penalty function. This method is repeated level by level until the leaves match the depth of the tree. 
By default, CatBoost uses symmetric trees ten times faster and gives better quality than non-symmetric trees.  
So far, the hassle why many do not consider using CatBoost is because of the slight difficulty in tuning the parameters to optimize the model for categorical features.
