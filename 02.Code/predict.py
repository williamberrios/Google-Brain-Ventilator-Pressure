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
import argparse
import wandb
DATA_PATH = '../01.Data'
DEBUG = False
os.environ['WANDB_SILENT'] = "True"


def parse_args():
    parser = argparse.ArgumentParser(description="Google Brain Kaggle")
    parser.add_argument("--folds", nargs="+",type=int, default=[0])
    parser.add_argument("--name",type=str, default='baseline')
    parser.add_argument("--dropout", nargs="+",type=float, default=0.0)
    parser.add_argument('--max_length', type=float, default=None)
    parser.add_argument('--checkpoints', nargs="+",type=int, default=[50])
    parser.add_argument('--ensemble_mode', type=str, default='median')
    args = parser.parse_args()
    return args


class Config:
    def __init__(self,args):
        # =========== General Parameters ========
        self.seed = 42
        self.logging = False
        self.run_folds       = args.folds
        self.max_length = args.max_length
        self.RC         = [None,None]
        self.init_weights = 'xavier'
        self.scaler_layer = False
        # ======== Model Parameters =============
        self.input_size  = -1
        self.hidden_size = 400
        self.num_layers  = 4
        self.dropout     = 0.0
        self.bidirectional = True
        self.logit_dim     = 128#50
        self.layer_normalization = False
        # ========= Training Parameters =========
        self.epochs         = 400
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
                                 'WD'  : 1e-6} #0
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


def predict(cfg,train_df,test_df,fold_predict,en_mode = 'mean',epoch = None):
    
    cfg.logging = False
    test_df['pressure'] = -1
    print('Data loaded')
    # Drop innecsary columns:
    try:
        test_df.drop(columns = ['R_indx','C_indx'],inplace = True)
        train_df.drop(columns = ['R_indx','C_indx'],inplace = True)
    except:
        print('already_dropped')
    try: 
        cfg.cols       = train_df.drop(columns = ['id','breath_id','pressure','u_out','fold','oof']).columns.to_list()
    except: 
        cfg.cols       = train_df.drop(columns = ['id','breath_id','pressure','u_out','fold']).columns.to_list()
    cfg.input_size = len(cfg.cols)
    # Scaling Dataset
    if cfg.max_length is not None:
        train_df = preprocess_max_length(train_df,cfg.max_length)
        test_df = preprocess_max_length(test_df,cfg.max_length)
    
    if cfg.sc_name is not None:
        scaler   = DataScaler(train_df,sc_name = cfg.sc_name,cols = cfg.cols)
        train_df = scaler.transform(train_df)
        test_df  = scaler.transform(test_df)
        print('Data Scaled')

    preds_kfold = 0
    train_df.loc[:,'oof'] = -1
    list_preds  = []
    for fold in fold_predict:
        print(f"***********************************************")
        print(f"**************** FOLD : {fold} *********************")
        print(f"***********************************************")
        cfg.fold        = fold
        print(cfg.output_dir,cfg.experiment_name,f'fold_{fold}')
        cfg.output_path = os.path.join(cfg.output_dir,cfg.experiment_name,f'fold_{fold}')
        # Training Dataset
        train     = train_df[train_df['fold']!=fold].reset_index(drop = True)
        # Valid Dataset
        valid     = train_df[train_df['fold']==fold]
        valid_index = valid.index.to_list()
        valid     = valid.reset_index(drop = True)
         # Model Part
        seed_everything(cfg.seed)
        model    = Model_LSTM(cfg)
        trainer  = Trainer(config = cfg,model = model) 
        if epoch is None:
            model_output = 'model.pt'
        else:
            model_output = f'model_epoch_{epoch}.pt'
        
        print(model_output)
        _, valid_preds = trainer.predict(valid,os.path.join(cfg.output_path,model_output))
        train_df.loc[valid_index,'oof'] = valid_preds.reshape(-1)
       
        _, test_preds = trainer.predict(test_df,os.path.join(cfg.output_path,model_output))
        list_preds.append(test_preds.reshape(-1))
        preds_kfold   += test_preds
    # Saving test preds
    if epoch is None:
        oof_output = 'oof_preds.csv'
    else:
        oof_output = f'oof_preds_epoch_{epoch}.csv'
        
    train_df[['id','breath_id','oof']].to_csv(os.path.join(cfg.output_dir,cfg.experiment_name,oof_output),index = False)
    
    if en_mode == 'mean':
        test_df['preds'] = (preds_kfold/len(fold_predict)).reshape(-1)
    elif en_mode == 'median': 
        test_df['preds'] = np.median(np.vstack(list_preds),axis=0)
       
    if epoch is None:
        test_output = 'test_preds.csv'
    else:
        test_output = f'test_preds_epoch_{epoch}_{en_mode}.csv'    
    test_df[['id','breath_id','preds']].to_csv(os.path.join(cfg.output_dir,cfg.experiment_name,test_output),index = False)     


if __name__ == '__main__':
    args = parse_args()
    cfg         = Config(args) 
    print('=== Loading Datasets ===') 
    train_df    = feat_eng(pd.read_csv(os.path.join(DATA_PATH,'train_folds.csv'),nrows = 80*100))   
    test_df     = feat_eng(pd.read_csv(os.path.join(DATA_PATH,'test.csv')))  
    print('=== Generating Predictions ===')
    for check in args.checkpoints:
        print(f"Prediction - {check} checkpoint")
        predict(cfg,train_df,test_df,cfg.run_folds,args.ensemble_mode,epoch = check) 
