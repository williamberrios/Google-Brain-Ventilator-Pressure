import pandas as pd
import os
import argparse
import numpy as np


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
        print('doing postprocesing')
        df = post_processing(df)
    df.to_csv('submission.csv',index = False)
    os.system(f'kaggle competitions submit -c ventilator-pressure-prediction -f submission.csv -m "{args.model_experiment}_{args.comment}"')
    if os.path.exists("submission.csv"):
        os.remove("submission.csv")


if __name__ == '__main__':
    main()
