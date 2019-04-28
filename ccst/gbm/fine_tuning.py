#coding=utf-8

'''
Created on Apr 28, 2019

@author: user
'''
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn import metrics

X_train = np.loadtxt('train_x.txt', delimiter='\t')
y_train = np.loadtxt('train_y.txt', delimiter='\t')
X_test = np.loadtxt('test_x.txt', delimiter='\t')
y_test = np.loadtxt('test_y.txt', delimiter='\t')

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

# 设置初始参数--不含交叉验证参数
print('设置参数')
params = {
          'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'metric': 'multi_error',
          'nthread':4,
          'num_class': 3,
          'learning_rate':0.1
          }
 
### 交叉验证(调参)
print('交叉验证')
min_err = float('1')
best_params = {}
 
# 准确率
print("调参1：提高准确率")
for num_leaves in range(5,100,5):
    for max_depth in range(3,8,1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth
 
        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=1,
                            nfold=5,
                            metrics=['multi_error'],
                            early_stopping_rounds=10,
                            verbose_eval=True
                            )
            
        mean_err = pd.Series(cv_results['multi_error-mean']).min()
        boost_rounds = pd.Series(cv_results['multi_error-mean']).idxmin()
            
        if mean_err <= min_err:
            min_err = mean_err
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth
if 'num_leaves' and 'max_depth' in best_params.keys():          
    params['num_leaves'] = best_params['num_leaves']
    params['max_depth'] = best_params['max_depth']
 
# 过拟合
print("调参2：降低过拟合")
for max_bin in range(5,256,10):
    for min_data_in_leaf in range(1,102,10):
            params['max_bin'] = max_bin
            params['min_data_in_leaf'] = min_data_in_leaf
            
            cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=1,
                                nfold=5,
                                metrics=['multi_error'],
                                early_stopping_rounds=10,
                                verbose_eval=True
                                )
                    
            mean_err = pd.Series(cv_results['multi_error-mean']).min()
            boost_rounds = pd.Series(cv_results['multi_error-mean']).idxmin()
 
            if mean_err <= min_err:
                min_err = mean_err
                best_params['max_bin']= max_bin
                best_params['min_data_in_leaf'] = min_data_in_leaf
if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
    params['min_data_in_leaf'] = best_params['min_data_in_leaf']
    params['max_bin'] = best_params['max_bin']
 
print("调参3：降低过拟合")
for feature_fraction in [0.6,0.7,0.8,0.9,1.0]:
    for bagging_fraction in [0.6,0.7,0.8,0.9,1.0]:
        for bagging_freq in range(0,50,5):
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = bagging_freq
            
            cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=1,
                                nfold=5,
                                metrics=['multi_error'],
                                early_stopping_rounds=10,
                                verbose_eval=True
                                )
                    
            mean_err = pd.Series(cv_results['multi_error-mean']).min()
            boost_rounds = pd.Series(cv_results['multi_error-mean']).idxmin()
 
            if mean_err <= min_err:
                min_err = mean_err
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq
 
if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
    params['feature_fraction'] = best_params['feature_fraction']
    params['bagging_fraction'] = best_params['bagging_fraction']
    params['bagging_freq'] = best_params['bagging_freq']
 
 
print("调参4：降低过拟合")
for lambda_l1 in [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]:
    for lambda_l2 in [1e-5,1e-3,1e-1,0.0,0.1,0.4,0.6,0.7,0.9,1.0]:
        params['lambda_l1'] = lambda_l1
        params['lambda_l2'] = lambda_l2
        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=1,
                            nfold=5,
                            metrics=['multi_error'],
                            early_stopping_rounds=10,
                            verbose_eval=True
                            )
                
        mean_err = pd.Series(cv_results['multi_error-mean']).min()
        boost_rounds = pd.Series(cv_results['multi_error-mean']).idxmin()
 
        if mean_err <= min_err:
            min_err = mean_err
            best_params['lambda_l1'] = lambda_l1
            best_params['lambda_l2'] = lambda_l2
if 'lambda_l1' and 'lambda_l2' in best_params.keys():
    params['lambda_l1'] = best_params['lambda_l1']
    params['lambda_l2'] = best_params['lambda_l2']
 
print("调参5：降低过拟合2")
for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    params['min_split_gain'] = min_split_gain
    
    cv_results = lgb.cv(
                        params,
                        lgb_train,
                        seed=1,
                        nfold=5,
                        metrics=['multi_error'],
                        early_stopping_rounds=10,
                        verbose_eval=True
                        )
            
    mean_err = pd.Series(cv_results['multi_error-mean']).min()
    boost_rounds = pd.Series(cv_results['multi_error-mean']).idxmin()
 
    if mean_err <= min_err:
        min_err = mean_err
        
        best_params['min_split_gain'] = min_split_gain
if 'min_split_gain' in best_params.keys():
    params['min_split_gain'] = best_params['min_split_gain']
 
print(best_params)
