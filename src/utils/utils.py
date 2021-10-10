import numpy as np
import os
import random
import torch
import inspect


# ## Seed Utils

# +
def seed_everything(seed=42):
    '''
    
    Function to put a seed to every step and make code reproducible
    Input:
    - seed: random state for the events 
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def worker_init_fn(worker_id):
    print(f"Seed for workers: {torch.initial_seed()}")
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -

# ## Configs utils

def get_attributes_config(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            pr[name] = value
    return pr
