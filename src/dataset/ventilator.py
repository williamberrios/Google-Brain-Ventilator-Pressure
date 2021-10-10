# +
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class VentilatorDataset(Dataset):
    def __init__(self, df,cols = []):
        assert len(cols)>0
        self.df   = df.groupby('breath_id').agg(list).reset_index().drop(columns = ['id'])
        self.cols = cols
        self.prepare_data()
                
    def __len__(self):
        return self.df.shape[0]
    
    def prepare_data(self):            
        self.pressures = np.array(self.df['pressure'].values.tolist())
        self.features  = np.array(self.df[self.cols].values.tolist()).transpose(0, 2, 1)
        self.u_out     = np.array(self.df['u_out'].values.tolist())
        #print(f'Target Shape: {self.pressures.shape}')
        #print(f'Features Shape: {self.features.shape}')
        
    def _get_breathid(self):
        return self.df.breath_id.values
    
    def __getitem__(self, idx):
        data = {
            "features" : torch.tensor(self.features[idx], dtype=torch.float),
            "u_out"    : torch.tensor(self.u_out[idx], dtype=torch.float),
            "pressure" : torch.tensor(self.pressures[idx], dtype=torch.float)
        }
        return data


# -

if __name__ == '__main__':
    df = pd.DataFrame({'breath_id' : [1,2],
                       'pressure'  : [1,2],
                       'u_in'      : [1,2],
                       'u_out'     : [1,2],
                       'id'        : ['id1','id2']})
    dataset = VentilatorDataset(df,cols = ['u_in'])
