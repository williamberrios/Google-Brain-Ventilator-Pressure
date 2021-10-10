from sklearn.preprocessing import RobustScaler, StandardScaler,MinMaxScaler 
import pandas as pd
class DataScaler():
    def __init__(self,df,sc_name = 'Robust',cols = []):
        self.df = df[cols]
        self.cols = cols
        self.scaler = self._getscaler(sc_name)
        self.fit(self.df)
        
    def fit(self,df):
        self.scaler.fit(self.df)
    
    def transform(self,data):
        data[self.cols] = self.scaler.transform(data[self.cols])
        return data
    
    def _getscaler(self,sc_name):
        if sc_name == 'Robust':
            return RobustScaler()
        elif sc_name == 'Standard':
            return StandardScaler()
        elif sc_name == 'MinMax':
            return MinMaxScaler()


if __name__ == '__main__':
    df = pd.DataFrame({'breath_id' : [1,2],
                       'pressure'  : [1,2],
                       'u_in'      : [1,2],
                       'u_out'     : [1,2],
                       'id'        : ['id1','id2']})
    scaler = DataScaler(df,'Robust',cols = ['u_in','u_out'])
    df = scaler.transform(df)
