import pandas as pd
import os
import argparse


# +
def parse_args():
    parser = argparse.ArgumentParser(description="Submmit 2 Kaggle")
    parser.add_argument('--model_experiment', type=str, default='None',
                        help='Name of the experiment (default: None)')
    parser.add_argument('--comment', type=str, default = '',
                        help='comment for submission')
    parser.add_argument('--postprocessing', type=bool, default = False,
                        help='Bolean for postprocessing')
    args = parser.parse_args()
    return args

def post_processing(df):
    train_df = pd.read_csv(os.path.join('../01.Data','train_folds.csv'))
    all_pressure = np.sort(train_df.pressure.unique() )
    press_min    = all_pressure[0]
    press_max    = all_pressure[-1]
    press_step   = ( all_pressure[1] - all_pressure[0] )
    df["pressure"] =np.round( (df.pressure - press_min)/press_step ) * press_step + press_min
    df["pressure"] = np.clip(df.pressure, press_min, press_max)
    return df


# -

def main():
    args = parse_args()
    test_df = pd.read_csv(os.path.join('../01.Data','test.csv'))[['id']]
    df = pd.read_csv(os.path.join('../03.SavedModels',args.model_experiment,'test_preds.csv'))\
           .drop(columns = ['breath_id'])\
           .rename(columns = {'preds':'pressure'}) 
    print(df.shape)
    if df.shape[0] !=test_df.shape[0]:
        df = test_df.merge(df,how = 'left',on = ['id'])
        df['pressure'] = df['pressure'].fillna(-1)
    if args.postprocessing:
        df = post_processing(df)
    df.to_csv('submission.csv',index = False)
    os.system(f'kaggle competitions submit -c ventilator-pressure-prediction -f submission.csv -m "{args.model_experiment}_{args.comment}"')
    if os.path.exists("submission.csv"):
        os.remove("submission.csv")


if __name__ == '__main__':
    main()

# python3.6 submit --model_experiment baseline_lstm_v4 --comment "num_layers_3_bidir_oof_0.022"
'''
import pandas as pd
import os
import numpy as np
test_df = pd.read_csv(os.path.join('../01.Data','test.csv'))[['id']]
df      = pd.read_csv(os.path.join('../03.SavedModels','baseline_lstm_v7','test_preds.csv'))\
            .drop(columns = ['breath_id'])\
            .rename(columns = {'preds':'pressure'}) 
print(df.shape)
if df.shape[0] !=test_df.shape[0]:
    df = test_df.merge(df,how = 'left',on = ['id'])
    df['pressure'] = df['pressure'].fillna(-1)

my_submission = df.copy()
# https://www.kaggle.com/mistag/pred-ventilator-lstm-model/data?select=submission.csv
kernel_sub    = pd.read_csv('kernel_subs/submission_2.csv')
my_submission['pressure'] = (my_submission['pressure'] + kernel_sub['pressure'])/2

def post_processing(df,press_min,press_step,press_max,col = 'pressure'):
    # ENSEMBLE FOLDS WITH MEDIAN AND ROUND PREDICTIONS
    df[col] =np.round( (df[col] - press_min)/press_step ) * press_step + press_min
    df[col] = np.clip(df[col], press_min, press_max)
    return df

train_df = pd.read_csv(os.path.join('../01.Data','train_folds.csv'))
all_pressure = np.sort( train_df.pressure.unique() )
press_min    = all_pressure[0]
press_max    = all_pressure[-1]
press_step   = ( all_pressure[1] - all_pressure[0] )
my_submission=  post_processing(my_submission,press_min,press_step,press_max)

my_submission.to_csv('submission_ensemble_with_free_kernels.csv',index = False)
'''
