#coding=utf-8

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import numpy as np
import pandas as pd



train_x = np.loadtxt('train_x.txt', delimiter='\t')
train_y = np.loadtxt('train_y.txt', delimiter='\t')
test_x = np.loadtxt('test_x.txt', delimiter='\t')
test_y = np.loadtxt('test_y.txt', delimiter='\t')

train_data = lgb.Dataset(train_x, label=train_y)
test_data = lgb.Dataset(test_x, label=test_y, reference=train_data)
param = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'num_leaves': 16,
    'num_trees': 100,
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_error',
    'max_bin': 255,
    'learning_rate': 0.05,
    'early_stopping': 10
}
# param = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'num_trees': 100,
#     'objective': 'multiclass',
#     'num_class': 3,
#     'metric': 'multi_error',
#      
#     'num_leaves': 95,
#     'lambda_l1': 0.3, 
#     'bagging_freq': 5, 
#     'lambda_l2': 0.7, 
#     'min_split_gain': 0.0,    
#     'min_data_in_leaf': 11,    
#     'max_bin': 255, 
#     'bagging_fraction': 0.9, 
#     'max_depth': 4, 
#     'feature_fraction': 1.0,
#     'early_stopping': 10
# }
num_round = 10

bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
bst.save_model('model.txt')
mybst = lgb.Booster(model_file='model.txt')  # init model
ypred_train = bst.predict(train_x)
ypred_test = bst.predict(test_x)
 
ypred_train = [list(x).index(max(x)) for x in ypred_train]
ypred_test = [list(x).index(max(x)) for x in ypred_test]
# print(ypred_train)
print('The train error rate of prediction is:',
       accuracy_score(train_y, ypred_train))
print('The test error rate of prediction is:',
       accuracy_score(test_y, ypred_test))

cv_results = lgb.cv(param, train_data, num_round, nfold=5)
print('best n_estimators:', len(cv_results['multi_error-mean']))
print('best cv score:', pd.Series(cv_results['multi_error-mean']).min())
