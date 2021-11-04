#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import os
import numpy as np
import sys
module_path = "../src"
if module_path not in sys.path:
    sys.path.append(module_path)
from utils.utils import seed_everything
from utils.scaler import DataScaler
from models.trainer import Trainer
from models.lstm import Model_LSTM
from sklearn.metrics import mean_absolute_error
from utils.utils import feat_eng,preprocess_max_length
import wandb
import argparse

DATA_PATH = '../01.Data'
DEBUG = False
os.environ['WANDB_SILENT']="True"


# %%
def train_fn(train,valid,cfg):
    # Call seed
    seed_everything(cfg.seed)
    model    = Model_LSTM(cfg)
    trainer  = Trainer(config = cfg,model = model)
    best_val_loss = trainer.fit(train,valid)
    print(f'Best Val Loss: {best_val_loss}')
    return trainer


# %%
def compute_mae_filtered(df,target = 'pressure',preds = 'oof'):
    metric_df  = df[df['u_out']!=1].reset_index(drop = True)
    mae_metric = mean_absolute_error(metric_df[target].values,metric_df[preds].values)
    del metric_df
    return mae_metric


# %%
def run_kfold(args):
    cfg       = Config(args)
    run_folds = cfg.run_folds 
    if DEBUG:
        train_df = feat_eng(pd.read_csv(os.path.join(DATA_PATH,args.dataset),nrows=80*100))
        test_df  = feat_eng(pd.read_csv(os.path.join(DATA_PATH,'test.csv'),nrows=80*100))
    else:
        
        train_df = feat_eng(pd.read_csv(os.path.join(DATA_PATH,args.dataset)))   
        test_df  = feat_eng(pd.read_csv(os.path.join(DATA_PATH,'test.csv')))
        
    if cfg.RC is not None:
        # Filter R and C:
        if cfg.RC[0] is not None:
            train_df = train_df[train_df['R_indx'].isin(cfg.RC[0])].reset_index(drop = True)
            test_df  = test_df[test_df['R_indx'].isin(cfg.RC[0])].reset_index(drop = True)
        if cfg.RC[1] is not None:
            train_df = train_df[train_df['C_indx'].isin(cfg.RC[1])].reset_index(drop = True)
            test_df  = test_df[test_df['C_indx'].isin(cfg.RC[1])].reset_index(drop = True)
            
    
    if cfg.max_length is not None:
        train_df = preprocess_max_length(train_df,cfg.max_length)
        test_df = preprocess_max_length(test_df,cfg.max_length)
    # Drop innecsary columns:
    train_df.drop(columns = ['R_indx','C_indx'],inplace = True)
    test_df.drop(columns = ['R_indx','C_indx'],inplace = True)
    test_df['pressure'] = -1
    cfg.cols       = train_df.drop(columns = ['id','breath_id','pressure','u_out','fold']).columns.to_list()
    cfg.input_size = len(cfg.cols)
    # Scaling Dataset
    if cfg.sc_name is not None:
        scaler   = DataScaler(train_df,sc_name = cfg.sc_name,cols = cfg.cols)
        train_df = scaler.transform(train_df)
        test_df  = scaler.transform(test_df)
    # Data to store
    preds_kfold = 0
    train_df.loc[:,'oof'] = -1
    for fold in run_folds:
        print(f"***********************************************")
        print(f"**************** FOLD : {fold} *********************")
        print(f"***********************************************")
        cfg.fold        = fold
        cfg.output_path = os.path.join(cfg.output_dir,cfg.experiment_name,f'fold_{fold}')
        # Training Dataset
        train     = train_df[train_df['fold']!=fold].reset_index(drop = True)
        
        # Valid Dataset
        valid     = train_df[train_df['fold']==fold]
        valid_index = valid.index.to_list()
        valid     = valid.reset_index(drop = True)
        # Trainer Part
        trainer   = train_fn(train,valid,cfg)
        # Valid Preds:
        _, valid_preds = trainer.predict(valid,os.path.join(cfg.output_path,'model.pt'))
        valid['oof'] = valid_preds.reshape(-1) 
        valid_metric = compute_mae_filtered(valid[['pressure','u_out','oof']])
        print('Valid Metric: ',valid_metric)
        trainer._log({'fold_metric':valid_metric})
        train_df.loc[valid_index,'oof'] = valid_preds.reshape(-1)
        if fold == run_folds[-1]:
            oof_metric = compute_mae_filtered(train_df[['pressure','u_out','oof']])
            trainer._log({'oof_metric':oof_metric})
        # Test Preds
        _, test_preds = trainer.predict(test_df,os.path.join(cfg.output_path,'model.pt'))
        preds_kfold   += test_preds
        if fold != run_folds[-1]:
            del trainer
        del train,valid
        
    # Saving test preds
    test_df['preds'] = (preds_kfold/len(run_folds)).reshape(-1)
    test_df[['id','breath_id','preds']].to_csv(os.path.join(cfg.output_dir,cfg.experiment_name,f'test_preds.csv'),index = False) 
    trainer._upload_df(name = 'preds', data = test_df[['id','breath_id','preds']])
    # Saving oof predictions
    train_df[['id','breath_id','oof']].to_csv(os.path.join(cfg.output_dir,cfg.experiment_name,f'oof_preds.csv'),index = False)
    trainer._upload_df(name = 'oof_train', data = train_df[['id','breath_id','oof']])
    trainer._finish()
    del trainer


# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Google Brain Kaggle")
    parser.add_argument("--folds", nargs="+",type=int, default=[0])
    parser.add_argument("--name",type=str, default='baseline')
    parser.add_argument("--dropout", nargs="+",type=float, default=0.0)
    parser.add_argument('--dataset', type=str, default='train_folds.csv')
    parser.add_argument('--max_length', type=float, default=None)
    
    args = parser.parse_args()
    return args


# %%
class Config:
    def __init__(self,args):
        # =========== General Parameters ========
        self.seed = 42
        self.logging = True
        self.run_folds       = args.folds
        self.max_length = args.max_length
        self.RC         = [None,None]
        self.init_weights = 'xavier'
        self.scaler_layer = False
        self.previous_path =  None#'../03.SavedModels/baseline_lstm_v15/'
        self.save_epoch    =  50 
        # ======== Model Parameters =============
        self.input_size  = -1
        self.hidden_size = 400
        self.num_layers  = 4
        self.dropout     = args.dropout
        self.bidirectional = True
        self.logit_dim     = 128
        self.layer_normalization = False
        # ========= Training Parameters =========
        self.epochs         = 200
        self.device         = 'cuda'
        self.lr             = 1e-3
        self.batch_size     = 2**9
        self.num_workers    = 72 
        self.sc_name        = 'Robust'
        # ======== Early stopping  =============
        self.early_stopping = 400
        self.mode           = 'min'
        # ======== Loss Parameters =============
        self.loss_params    = {'name':'MAE_FILTERED'}        
        # ======== Optimizer Parameters ========
        self.optimizer_params = {'name':'Adam',
                                 'WD'  : 1e-6} 

        self.scheduler_params = {'name'     : 'CosineAnnealingWarmRestarts',
                                 'step_on'  : 'epoch',
                                 'step_metric': 'valid_loss',
                                 'min_lr':1e-6,
                                 'T_0': 50}
        # ======= Logging and Saving Parameters ===
        self.project_name    = 'Ventilator-Kaggle'
        self.experiment_name = args.name
        self.fold            = None
        self.output_dir      = '../03.SavedModels' # Relative to trainer path


# %%
if __name__ == '__main__':
    args = parse_args()
    run_kfold(args)

