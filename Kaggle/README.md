# Kaggle Jupyter技巧总结

> 利用pandas + seaborn来处理csv的数据，快捷而且直观。
> [Comprehensive data exploration with Python | Kaggle](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)

## 环境配置
用下面的方式来手动注册当前的conda env到jupyter里面。

```bash
conda install ipykernel
python -m ipykernel install --user --name cornll-tech --display-name "Python (Cornell-Tech)"
```

## 可视化csv
> 可以在分析的时候帮助*发现重点feature*
> 只对numerical的field起作用。

```python
#histogram
sns.distplot(df_train['SalePrice']);

#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# boxplot
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# Correlation matrix (heatmap style)
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})

# pairwise scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();
```

## 数据预处理
> 针对categorical data -> one-hot encoding。
> 以及针对缺失的数据的处理：
> 	- 补全
> 	- 摒弃
>  
> pandas的dataframe的操作思路和numpy很一致

```python
# change column datatype. note missing fields. 
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

# reomve entry with conditions. (such as missing values.)
df = df[df.TotalCharges != ' ']

# drop columns.
df = df.drop(columns=['customerID'])

# create one-hot encoding for all categorical fields. 
encoded_df = pd.get_dummies(df)

#  get labels from data and process.
Y = df.pop('output')
Y = (Y == 'Yes').astype(int)

# set display option to show more columns. 
pd.set_option('display.max_columns', 50) 
encoded_df.head()
```

## Models
> 主要借助了 sklearn pipeline.
>  
> numerical data主要需要做normalization。

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# normalize columns of type float: 'MonthlyCharges' and 'TotalCharges'
num_ss_step = ('ss', StandardScaler())
num_steps = [num_ss_step]

num_pipe = Pipeline(num_steps)
num_cols = ['MonthlyCharges', 'TotalCharges']
num_transformers = [('num', num_pipe, num_cols)]
ct = ColumnTransformer(transformers=num_transformers)
X_num_transformed = ct.fit_transform(encoded_df)
X_num_transformed.shape # (7032, 2)
```


### 套用简单模型。尤其对于高维稀疏数据的regression问题。
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

models = [
    ('LR'   , LogisticRegression()),
    ('LDA'  , LinearDiscriminantAnalysis()),
    ('KNN'  , KNeighborsClassifier()),
    ('CART' , DecisionTreeClassifier()),
    ('NB'   , GaussianNB()),
    ('SVM'  , SVC(probability=True)),
    ('AB'   , AdaBoostClassifier()),
    ('GBM'  , GradientBoostingClassifier()),
    ('RF'   , RandomForestClassifier()),
    ('ET'   , ExtraTreesClassifier())
]

def run_models(x, y, models):
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds, random_state=123)
        cv_results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    return names, results

names, results = run_models(X, Y, models)
"""
得到的结果:
LR: 0.803470 (0.009425)
LDA: 0.797354 (0.011074)
KNN: 0.772755 (0.013865)
CART: 0.719289 (0.018143)
NB: 0.694681 (0.019061)
SVM: 0.798207 (0.010633)
AB: 0.805602 (0.010468)
GBM: 0.804893 (0.013025)
RF: 0.781855 (0.010472)
ET: 0.769192 (0.016778)
"""
```

### 通过 GridSearch 来找到对应classifier的最优参数
```python
from sklearn.model_selection import GridSearchCV
"""
Find best parameters for each classifiers.
Currently best classifiers:
    - LR
        {'lr__C': 0.01, 'lr__penalty': 'l2'}
        accuracy : 0.804749715585893
    - AB
        {'ab__learning_rate': 0.1, 'ab__n_estimators': 170}
        accuracy : 0.8053185437997725
    - GBM
        {'gb__learning_rate': 0.1, 'gb__n_estimators': 70}
        accuracy : 0.8065984072810012
    - KNN
        {'knn__n_neighbors': 23}
        accuracy : 0.786547212741752
    - SVM
        
"""

# ml_pipe = Pipeline([('ridge', Ridge())]) # 0.276
ml_pipe = Pipeline([('svm', SVC(probability=True)),])
ml_pipe.fit(X, Y)
ml_pipe.score(X, Y)

param_grid = {"svm__C": range(0.5, 2, 0.4), "svm__kernel": ['linear', 'rbf']}
gs = GridSearchCV(ml_pipe, param_grid, cv=10)
gs.fit(X, Y)
print("tuned hpyerparameters :(best parameters) ",gs.best_params_)
print("accuracy :",gs.best_score_)
```

### Ensemble

找到最优的参数之后，可以来Ensemble.
第一种是普通的Voting。

```python
from sklearn.ensemble import VotingClassifier
"""
Ensemble from the best models. 
Basic Voting.
"""
param = {'C': 0.01, 'penalty': 'l2'}
model1 = LogisticRegression(**param)

param = {'learning_rate': 0.1, 'n_estimators': 170}
model2 = AdaBoostClassifier(**param)

param = {'learning_rate': 0.1, 'n_estimators': 70}
model3 = GradientBoostingClassifier(**param)

estimators = [('LR', model1), ('AB', model2), ('GB', model3)]

kfold = StratifiedKFold(n_splits=10, random_state=123)
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold, scoring='accuracy')
results.mean()
```

第二种是Stacking。meta learner。
```python
"""
Stacking Model using lib: mlens.
"""

def get_models():
    """Generate a library of base learners."""
    param = {'C': 0.01, 'penalty': 'l2'}
    model1 = LogisticRegression(**param)

    param = {'learning_rate': 0.1, 'n_estimators': 170}
    model2 = AdaBoostClassifier(**param)

    param = {'learning_rate': 0.1, 'n_estimators': 70}
    model3 = GradientBoostingClassifier(**param)

    param = {'n_neighbors': 23}
    model4 = KNeighborsClassifier(**param)

    param = {'C': 1.7, 'kernel': 'rbf', 'probability':True}
    model5 = SVC(**param)

    param = {'criterion': 'gini', 'max_depth': 3, 'max_features': 2, 'min_samples_leaf': 3}
    model6 = DecisionTreeClassifier(**param)

    model7 = GaussianNB()

    model8 = RandomForestClassifier()

    model9 = ExtraTreesClassifier()

    models = {'LR':model1, 'ADA':model2, 'GB':model3,
              'KNN':model4, 'SVM':model5, 'DT':model6,
              'NB':model7, 'RF':model8,  'ET':model9
              }

    return models

base_learners = get_models()
meta_learner = GradientBoostingClassifier(
    n_estimators=1000,
    loss="exponential",
    max_features=6,
    max_depth=3,
    subsample=0.5,
    learning_rate=0.001, 
    random_state=123
)

from mlens.ensemble import SuperLearner

# Instantiate the ensemble with 10 folds
sl = SuperLearner(
    folds=10,
    random_state=123,
    verbose=2,
    backend="multiprocessing"
)

# Add the base learners and the meta learner
sl.add(list(base_learners.values()), proba=True) 
sl.add_meta(meta_learner, proba=True)

# Train the ensemble
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test =train_test_split(X,Y, test_size=0.2, random_state=0)

sl.fit(X_train, Y_train)

# Predict the test set
p_sl = sl.predict_proba(X_test)

pp = []
for p in p_sl[:, 1]:
    if p>0.5:
        pp.append(1.)
    else:
        pp.append(0.)

print("\nSuper Learner Accuracy score: %.8f" % (Y_test== pp).mean())
```

## 参考资料
🌟 大部分ensemble的方法 [A Complete ML Pipeline Tutorial (ACU ~ 86%) | Kaggle](https://www.kaggle.com/pouryaayria/a-complete-ml-pipeline-tutorial-acu-86)

🌟 结合pandas和scikit learn的workflow [From Pandas to Scikit-Learn — A new exciting workflow](https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62)

grid search [Grid Search with Logistic Regression | Kaggle](https://www.kaggle.com/enespolat/grid-search-with-logistic-regression)
















