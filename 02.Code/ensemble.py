import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
MODEL_PATH = '../03.SavedModels'
DATA_PATH  = '../01.Data/'
MODEL_NAMES = {'baseline_lstm_v17':[None,600,550],'baseline_lstm_v15':[None],'baseline_lstm_v18':[None,200]}
MODE_ENSEMBLE = 'median'
PREDICTION_ROUNDED = False
METHOD_ROUNDING = 2


# +
def post_processing(df,col = 'pressure'):
    df[col] = df[col].apply(find_nearest)
    return df

def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return sorted_pressures[-1]
    elif insert_idx == 0:
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return lower_val if abs(lower_val - prediction) < abs(upper_val - prediction) else upper_val

def calculate_metric(df,pred = 'oof',target = 'pressure'):
    tmp = df.loc[df['u_out']==0].copy().reset_index(drop = True)
    return mean_absolute_error(tmp[pred],tmp[target])


# -

# %%time
train_df = pd.read_csv(os.path.join(DATA_PATH,'train_folds.csv'))
unique_pressures = train_df["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)

# +
## =============================================
## Loading test predictions and OOF predictions
## =============================================

oof_list   = []
preds_list = []
oof_values = []
train_df = pd.read_csv('../01.Data/train_folds.csv')[['id','pressure','u_out']]
test_df  = pd.read_csv('../01.Data/test.csv')[['id','u_out']]

for name in tqdm(MODEL_NAMES.keys()):
    for checkpoint in MODEL_NAMES[name]:
        if checkpoint is not None:
            oof_df  = pd.read_csv(os.path.join(MODEL_PATH,name,f'oof_preds_epoch_{checkpoint}.csv')).sort_values(by = ['id'])
            pred_df = pd.read_csv(os.path.join(MODEL_PATH,name,f'test_preds_epoch_{checkpoint}_median.csv')).sort_values(by = ['id']) 
            oof_df = train_df.merge(oof_df,how = 'left',on = ['id']).sort_values(by = ['id']).fillna(-1)
            pred_df = test_df.merge(pred_df,how = 'left',on = ['id']).sort_values(by = ['id']).fillna(-1)
            if PREDICTION_ROUNDED:
                pred_df = post_processing(pred_df,col = 'preds')
        else:
            oof_df  = pd.read_csv(os.path.join(MODEL_PATH,name,'oof_preds.csv')).sort_values(by = ['id'])
            pred_df = pd.read_csv(os.path.join(MODEL_PATH,name,'test_preds.csv')).sort_values(by = ['id']) 
            oof_df = train_df.merge(oof_df,how = 'left',on = ['id']).sort_values(by = ['id']).fillna(-1)
            pred_df = test_df.merge(pred_df,how = 'left',on = ['id']).sort_values(by = ['id']).fillna(-1)
            if PREDICTION_ROUNDED:
                pred_df = post_processing(pred_df,col = 'preds')
            
        print(f"========== OOF METRIC: {name} - checkpoint[{checkpoint}] ========")
        mae_original = calculate_metric(oof_df,pred = 'oof',target = 'pressure')
        if PREDICTION_ROUNDED:
            oof_df = post_processing(oof_df,col = 'oof')
            mae_post = calculate_metric(oof_df,pred = 'oof',target = 'pressure')
            print(f"MAE-ORIGIN:{mae_original}, MAE_POST: {mae_post}")
            oof_values.append(mae_post)
        else:
            print(f"MAE-ORIGIN:{mae_original}")
            oof_values.append(mae_original)
            
        oof_list.append(oof_df.oof.values) # With postprocessed values
        preds_list.append(pred_df.preds.values)
        del oof_df,pred_df

# +
if MODE_ENSEMBLE == 'mean':
    oof_ens   = np.mean(np.vstack(oof_list),axis = 0)
    preds_ens = np.mean(np.vstack(preds_list),axis = 0)
    
elif MODE_ENSEMBLE == 'median':
    oof_ens   = np.median(np.vstack(oof_list),axis = 0)
    preds_ens = np.median(np.vstack(preds_list),axis = 0)

print(f"{'='*5} ENSEMBLE {MODE_ENSEMBLE} {'='*5}")

train_df['oof'] = oof_ens
tmp = train_df.loc[train_df['u_out']==0].reset_index(drop = True)
mae_original = mean_absolute_error(tmp.oof,tmp.pressure)
tmp = post_processing(tmp,col = 'oof')
mae_post = mean_absolute_error(tmp.oof,tmp.pressure)
print(f"MAE-ORIGIN:{mae_original}, MAE_POST: {mae_post}")
# -

test_df['pressure'] = preds_ens
test_df = test_df[['id','pressure']]
test_df      =  post_processing(test_df,col = 'pressure')
test_df.to_csv('final_submission.csv',index = False)
