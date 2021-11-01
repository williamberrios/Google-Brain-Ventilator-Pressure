# -*- coding: utf-8 -*-
# +
import pandas as pd
import lightgbm as lgbm
import numpy as np
import os 
import time
import math
from sklearn.metrics import mean_absolute_error

def Training_Lightgbm(df,params,fold_column = 'fold',target_column = 'target',cat_vars = [],metric = 'MAE',early_stopping = 200,max_boost_round = 8000):
    tr_metric  = []
    val_metric = []
    importances = pd.DataFrame()
    importances['Features'] = df.drop(columns  = [fold_column,target_column]).columns[:]
    models = []
    oof = np.zeros((len(df)))
    start = int(time.time() * 1000)
    for i in range(len(df[fold_column].unique())):
      # Get the train and Valid sets
        df_tr  = df[df[fold_column]!=i].reset_index(drop  = True)
        df_val = df[df[fold_column]==i].reset_index(drop  = True)
        # Split by independent and dependent variables
        X_train, y_train = df_tr.drop(columns  = [fold_column,target_column]) , df_tr[target_column]
        X_valid, y_valid = df_val.drop(columns  = [fold_column,target_column]), df_val[target_column]
        feature_list = X_train.columns.to_list()
        print(f"Columns: {X_train.columns.to_list()}")
        # Index for categorical variable if exist
        if cat_vars:
            features = [x for x in X_train.columns]
            cat_ind = [features.index(x) for x in cat_vars]
        else:
            cat_ind = []
        print('Cat index:', cat_ind)
        # Create lgbm Dataset 
        lgbm_train = lgbm.Dataset(X_train, label = y_train,categorical_feature= cat_ind)
        lgbm_eval  = lgbm.Dataset(X_valid, y_valid, reference = lgbm_train,categorical_feature=cat_ind)

        # Training lgbm model
        print('---------- Training fold Nº {} ----------'.format(i+1))
        lgbm_model = lgbm.train(params,lgbm_train,num_boost_round = max_boost_round,early_stopping_rounds = early_stopping, verbose_eval=50, categorical_feature=cat_ind,valid_sets=[lgbm_train,lgbm_eval]) 
        # Initialize Importances
        name = 'importance_'+str(i)
        importances[name]= lgbm_model.feature_importance(importance_type = 'gain')
        # Saving Model
        models.append(lgbm_model)
        # Saving oof predictions
        valid_idx = df[df[fold_column] == i].index
        # Evaluating Metrics for train and Validation
        y_train_pred = lgbm_model.predict(X_train,num_iteration=lgbm_model.best_iteration)
        tr_metric    += [mean_absolute_error(y_train, y_train_pred)]

        y_valid_pred   = lgbm_model.predict(X_valid,num_iteration=lgbm_model.best_iteration)
        val_metric    += [mean_absolute_error(y_valid, y_valid_pred)]
        oof[valid_idx] = y_valid_pred
        print(f"Train {metric}: {tr_metric[-1]}        Valida {metric}: {val_metric[-1]}")
    end = int(time.time() * 1000)
    results = pd.DataFrame({'Model_Name'           : ['Lgbm Model'],
                          f'Mean Valid {metric}' : [np.mean(val_metric)],
                          f'Std Valid {metric}'  : [np.std(val_metric)],
                          f'Mean Train {metric}' : [np.mean(tr_metric)],
                          f'Std Train {metric}'  : [np.std(tr_metric)],
                          f'OOF {metric}'        : [mean_absolute_error(df[target_column].values, oof)],
                          f'Diff {metric}'       : [np.mean(tr_metric) - np.mean(val_metric)],
                          'Time'                 : [str(end-start) + ' s']})
    print(f'OOF {metric}: {mean_absolute_error(df[target_column].values, oof)} ')                         
    return results,models,importances,oof,feature_list
# -


