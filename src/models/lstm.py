import torch 
import torch.nn as nn
import pandas as pd
import numpy as np

'''
train_df = pd.read_csv('../../01.Data/train.csv')
all_pressure = sorted(train_df.pressure.unique())
PRESSURE_MIN = np.min(all_pressure)
PRESSURE_MAX = np.max(all_pressure)
PRESSURE_STEP = all_pressure[1] - all_pressure[0]
'''


# +
class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class ScaleLayer(nn.Module):
    def __init__(self):
        super(ScaleLayer, self).__init__()
        self.min = PRESSURE_MIN
        self.max = PRESSURE_MAX
        self.step = PRESSURE_STEP
        self.my_round_func = my_round_func()

    def forward(self, inputs):
        steps = inputs.add(-self.min).divide(self.step)
        int_steps = self.my_round_func.apply(steps)
        rescaled_steps = int_steps.multiply(self.step).add(self.min)
        clipped = torch.clamp(rescaled_steps, self.min, self.max)
        return clipped



# +

class Model_LSTM(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.lstm = nn.LSTM(input_size  = cfg.input_size,
                            hidden_size = cfg.hidden_size,
                            num_layers  = cfg.num_layers,
                            bias        = True,
                            batch_first = True,
                            dropout     = cfg.dropout,
                            bidirectional = cfg.bidirectional)
            
        if cfg.layer_normalization == False:
            print('No layer Normalization at the head')
            self.logits = nn.Sequential(nn.Linear(cfg.hidden_size * self._get_lstmmul(cfg), cfg.logit_dim),
                                        nn.ReLU(),
                                        nn.Linear(cfg.logit_dim,1))
        elif cfg.layer_normalization:
            print('Yes/layer Normalization at the head')
            self.logits = nn.Sequential(nn.Linear(cfg.hidden_size * self._get_lstmmul(cfg),cfg.hidden_size * self._get_lstmmul(cfg)),
                                        nn.LayerNorm(cfg.hidden_size * self._get_lstmmul(cfg)),   
                                        nn.ReLU(),
                                        nn.Linear(cfg.hidden_size * self._get_lstmmul(cfg),1))
        
        if cfg.scaler_layer:
            self.logits = nn.Sequential(self.logits,
                                        ScaleLayer())
            
        if cfg.init_weights is not None:
            self._init_weights_(cfg.init_weights)
    
    def _get_lstmmul(self,cfg):
        if cfg.bidirectional:
            return 2
        else:
            return 1
        
    def forward(self,x):
        x, _ = self.lstm(x)
        x    = self.logits(x)
        return x.squeeze()
    
    def _init_weights_(self,name):
        for n, m in self.named_modules():
            if name == 'xavier':
                if isinstance(m, nn.LSTM):
                    print(f'init {m}')
                    for param in m.parameters():
                        if len(param.shape) >= 2:
                            nn.init.xavier_normal_(param.data)
                        else:
                            nn.init.normal_(param.data)

                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.zeros_(m.bias)   
                
            elif name == 'orthogonal':
                if isinstance(m, nn.LSTM):
                    print(f'init {m}')
                    for param in m.parameters():
                        if len(param.shape) >= 2:
                            nn.init.orthogonal_(param.data)
                        else:
                            nn.init.normal_(param.data)
                elif isinstance(m, nn.GRU):
                    print(f"init {m}")
                    for param in m.parameters():
                        if len(param.shape) >= 2:
                            init.orthogonal_(param.data)
                        else:
                            init.normal_(param.data)
            else:
                raise Exception('Initialization not implemented')


# -

if __name__ == '__main__':
    class Config(object): 
        input_size  = 5
        hidden_size = 300
        num_layers  = 4
        dropout     = 0
        bidirectional = True
        logit_dim     = 100
        init_weights  = True
        layer_normalization = False
        init_weights  = 'xavier'
        scaler_layer  = True 
    cfg   = Config()
    model =  Model_LSTM(cfg)
    inputs = torch.randn(5,10,cfg.input_size)
    print(f'Output shape: {model(inputs).shape}')
    print(model)
