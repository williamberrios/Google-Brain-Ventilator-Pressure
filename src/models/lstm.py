import torch 
import torch.nn as nn
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
            
        self.logits = nn.Sequential(nn.Linear(cfg.hidden_size * self._get_lstmmul(cfg), cfg.logit_dim),
                                    nn.ReLU(),
                                    nn.Linear(cfg.logit_dim,1))
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


if __name__ == '__main__':
    class Config(object): 
        input_size  = 5
        hidden_size = 300
        num_layers  = 4
        dropout     = 0
        bidirectional = True
        logit_dim     = 100
        init_weights  = True
    cfg   = Config()
    model =  Model_LSTM(cfg)
    inputs = torch.randn(5,10,cfg.input_size)
    print(f'Output shape: {model(inputs).shape}')

