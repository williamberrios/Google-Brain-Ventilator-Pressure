# +
import pandas as pd
import numpy as np
from sklearn import preprocessing

class ClassicEncoding():
    
    def __init__(self,columns,params = None,name_encoding = 'LE'):
        """
        name_encoding : OHE,LE
        """
        self.name_encoding = name_encoding
        self.columns   = columns

        if params is not None:
            self.params = params
        else:
            if name_encoding == 'LE':
                self.params  = {'drop_original':True,
                                'missing_new_cat':True}
            elif name_encoding == 'OHE':
                self.params  = {'dummy_na':True,
                                        'drop_first':True,
                                        'drop_original':True}
        
    def one_hot_encoding(self,df,dummies_col=[], dummy_na=True ,drop_first = True, drop_original = True):
        '''
        Function to get the one_hot_encoding of a variable
        Input:
        -df            : Dataframe al cual se le aplica one hot encoding
        -dummies_col   : Variables categoricas
        -drop_cols     : Variables a eliminar
        -drop first    : Flag para indicar si se elimina una variable del one hot encoding
        -drop_original : Flag para indicar si eliminar las columnas originales
        '''
        df_ = df.copy()
        df_cols = df[dummies_col].copy()
        df_cols = pd.get_dummies(data = df_cols, columns = dummies_col, dummy_na = dummy_na, drop_first = drop_first)
        drop_cols_unique = [c for c in df_cols.columns if df_cols[c].nunique() <= 1]
        df_cols.drop(columns = drop_cols_unique, inplace = True)
        df_ = pd.concat([df_,df_cols],axis = 1)
        if drop_original:
            df_.drop(columns = dummies_col, inplace = True)

        return df_

    def label_encoding(self,df,label_cols = [], drop_original = True, missing_new_cat = True):
        '''
        Function to get the encoding Label of a variable
        Input:
        -df         : Dataframe al cual se le aplica one hot encoding
        -label_cols : Variables categoricas
        '''
        from sklearn import preprocessing
        df_     = df.copy()
        df_cols = df[label_cols].copy().rename(columns = { i : 'label_' + i for i in label_cols})
        dict_le ={}
        if missing_new_cat:
            print('Mode: Missing as new category')
            for i in df_cols.columns:
                le = preprocessing.LabelEncoder()
                print('Label Encoding: ',i)
                df_cols[i] = df_cols[i].astype('str')
                le.fit(df_cols[i])
                df_cols[i] = le.transform(df_cols[i])
                var_name = i
                dict_le[var_name] = le
        else:
            print('Mode: Missing as -1')
            for i in df_cols.columns:
                df_cols[i] = df_cols[i].fillna('NaN')
                df_cols[i] = df_cols[i].astype('str')
                le = preprocessing.LabelEncoder()
                print('Label Encoding: ',i)
                a = df_cols[i][df_cols[i]!='NaN']
                b = df_cols[i].values
                le.fit(a)
                b[b!='NaN']  = le.transform(a)
                df_cols[i] = b
                df_cols[i] = df_cols[i].replace({'NaN':-1})
                var_name = i
                dict_le[var_name] = le

        df_ = pd.concat([df_ , df_cols], axis = 1) 
        if drop_original:
            df_.drop(columns = label_cols, inplace = True)
        return df_,dict_le

    def apply_label_encoder(self,df,dict_label_encoder,drop_original = True, missing_new_cat = True):
        from sklearn import preprocessing
        df_     = df.copy()
        label_cols = [i[6:] for i in list(dict_label_encoder.keys())]
        df_cols = df[label_cols].copy().rename(columns = { i : 'label_' + i for i in label_cols})
        if missing_new_cat:
            print('Mode: Missing as new category')
            for i in df_cols.columns:
                print('Applying Label Encoding: ',i)
                df_cols[i] = df_cols[i].astype('str')
                le = dict_label_encoder[i]
                df_cols[i] = le.transform(df_cols[i])

        else:
            print('Mode: Missing as -1')
            for i in df_cols.columns:
                df_cols[i] = df_cols[i].fillna('NaN')
                df_cols[i] = df_cols[i].astype('str')
                print('Applying Label Encoding: ',i)
                a = df_cols[i][df_cols[i]!='NaN']
                b = df_cols[i].values
                le = dict_label_encoder[i]
                b[b!='NaN']  = le.transform(a)
                df_cols[i] = b
                df_cols[i] = df_cols[i].replace({'NaN':-1})

        df_ = pd.concat([df_ , df_cols], axis = 1) 
        if drop_original:
            df_.drop(columns = label_cols, inplace = True)
        return df_  

    
    def fit_transform(self,df):
        if self.name_encoding == 'OHE':
            return self.one_hot_encoding(df,
                                    self.columns, 
                                    self.params['dummy_na'],
                                    self.params['drop_first'], 
                                    self.params['drop_original'])
            
            
        elif self.name_encoding == 'LE':
            df,self.dict_le  = self.label_encoding(df,
                                          self.columns, 
                                          self.params['drop_original'], 
                                          self.params['missing_new_cat'])
            
            return df
            
        else: 
            print('None encoding constructed')
    
    def transform(self,df):
        if self.name_encoding == 'OHE':
            return self.one_hot_encoding(df,
                                    self.columns, 
                                    self.params['dummy_na'],
                                    self.params['drop_first'], 
                                    self.params['drop_original'])
            
        elif self.name_encoding == 'LE':
            df   = self.apply_label_encoder(df,
                                       self.dict_le,
                                       self.params['drop_original'], 
                                       self.params['missing_new_cat'])
            
            return df
            
        else: 
            print('None encoding constructed')
