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
import wandb
DATA_PATH = '../01.Data'
DEBUG = False
os.environ['WANDB_SILENT'] = "True"


def feat_eng(df):
    # Add Feature engineering df:
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)
    
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)
    return df    


class Config():
    # =========== General Parameters ========
    seed = 42
    logging = False
    run_folds       = [0,1,2,3,4]
    max_length = None
    RC         = [None,None]
    init_weights = 'xavier'
    # ======== Model Parameters =============
    input_size  = -1
    hidden_size = 400
    num_layers  = 4
    dropout     = 0.0
    bidirectional = True
    logit_dim     = 128#50
    # ========= Training Parameters =========
    epochs         = 300
    device         = 'cuda'
    lr             = 1e-3
    batch_size     = 2**9
    num_workers    = 72 
    sc_name        = 'Robust'
    # ======== Early stopping  =============
    early_stopping = 20 #10
    mode           = 'min'
    # ======== Loss Parameters =============
    loss_params    = {'name':'MAE_FILTERED'}        
    # ======== Optimizer Parameters ========
    optimizer_params = {'name':'Adam',
                        'WD'  : 1e-6} #0

    # ======= Scheduler Parameters =========
    # Mode: ['batch','epoch']
    #scheduler_params = {'name'     : 'Plateu',
    #                    'step_on'  : 'epoch',
    #                    'patience' :  8,#5
    #                    'step_metric': 'valid_loss'}
    scheduler_params = {'name'     : 'CosineAnnealingLR',
                        'step_on'  : 'epoch',
                        'step_metric': None,
                        'min_lr':1e-6,
                        'T_max': 300}
    # ======= Logging and Saving Parameters ===
    project_name    = 'Ventilator-Kaggle'
    experiment_name = 'baseline_lstm_v7'
    fold            = None
    output_dir      = '../03.SavedModels' # Relative to trainer path


# +
def preprocess_max_length(data,max_length):
    data['row_order'] = data.groupby(['breath_id']).cumcount()+1    
    data              = data[data['row_order']<=max_length].reset_index(drop = True)\
                                                           .sort_values(by = ['id'],ascending = True)
    data.drop(columns = ['row_order'],inplace = True)
    return data

def predict(fold_predict,en_mode = 'mean'):
    cfg         = Config() 
    cfg.logging = False
    train_df = feat_eng(pd.read_csv(os.path.join(DATA_PATH,'train_folds.csv')))   
    test_df  = feat_eng(pd.read_csv(os.path.join(DATA_PATH,'test.csv')))
    test_df['pressure'] = -1
    print('Data loaded')
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
    # Data to store
    preds_kfold = 0
    train_df.loc[:,'oof'] = -1
    list_preds  = []
    for fold in fold_predict:
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
         # Model Part
        seed_everything(cfg.seed)
        model    = Model_LSTM(cfg)
        trainer  = Trainer(config = cfg,model = model) 
        
        _, valid_preds = trainer.predict(valid,os.path.join(cfg.output_path,'model.pt'))
        train_df.loc[valid_index,'oof'] = valid_preds.reshape(-1)
       
        _, test_preds = trainer.predict(test_df,os.path.join(cfg.output_path,'model.pt'))
        list_preds.append(test_preds.reshape(-1))
        preds_kfold   += test_preds
    # Saving test preds
    train_df[['id','breath_id','oof']].to_csv(os.path.join(cfg.output_dir,cfg.experiment_name,f'oof_preds.csv'),index = False)
    if en_mode == 'mean':
        test_df['preds'] = (preds_kfold/len(fold_predict)).reshape(-1)
    elif en_mode == 'median': 
        test_df['preds'] = np.median(np.vstack(list_preds),axis=0)
   
    
    test_df[['id','breath_id','preds']].to_csv(os.path.join(cfg.output_dir,cfg.experiment_name,f'test_preds.csv'),index = False) 
    return test_df
# -

if __name__ == '__main__':
    test_df = predict([0,1,2,3,4],'mean')


# +
# POST PROCESSING
def post_processing(df,press_min,press_step,press_max):
    # ENSEMBLE FOLDS WITH MEDIAN AND ROUND PREDICTIONS
    df["preds"] =np.round( (df.preds - press_min)/press_step ) * press_step + press_min
    df["preds"] = np.clip(df.preds, press_min, press_max)
    return df

train_df = pd.read_csv(os.path.join(DATA_PATH,'train_folds.csv'))
all_pressure = np.sort( train_df.pressure.unique() )
press_min    = all_pressure[0]
press_max    = all_pressure[-1]
press_step   = ( all_pressure[1] - all_pressure[0] )
test_df      =  post_processing(test_df,press_min,press_step,press_max)
test_df      = test_df[['id','preds']].rename(columns = {'preds':'pressure'})
test_df.to_csv('submission.csv',index = False)

# +
#test_all_df  = pd.read_csv(os.path.join(DATA_PATH,'test.csv'))[['id','u_out']]
#test_all_df  = test_all_df.merge(test_df[['id','preds']],on = ['id'],how = 'left')
#test_all_df  = test_all_df.fillna(-1)

# +
#df = test_all_df[['id','preds']].rename(columns = {'preds':'pressure'})
#df.to_csv('submission.csv',index = False)
