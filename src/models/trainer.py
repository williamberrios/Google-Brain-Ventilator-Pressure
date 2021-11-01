import pandas as pd
import os
import numpy as np
import sys
import torch 
import torch.nn as nn
import wandb
import time
import gc
from tqdm import tqdm
module_path = ["../../src"]
for module in module_path:
    if module not in sys.path:
        sys.path.append(module)
from sklearn.metrics import mean_absolute_error
from utils.AverageMeter import AverageMeter
from utils.EarlyStopping import EarlyStopping
from utils.utils import seed_everything,get_attributes_config
from dataset.ventilator import VentilatorDataset
from utils.loss import MAE_FILTERED


class Trainer:
    def __init__(self,
                 config = None,
                 model  = None):
        self.config = config
        if self.config.logging:
            self.run = self._start_group_wandb()
        # Calculating the Model, Scheduler, Optimizer and Loss
        self.model     = model.to(self.config.device)
        self.criterion = self._fetch_loss()
        self.optimizer = self._fetch_optimizer()
        self.scheduler = self._fetch_scheduler()

    def train_fn(self,train_loader):
        # Model: train-mode
        self.model.train()
        # Initialize object Average Meter
        losses = AverageMeter()
        tk0 = tqdm(train_loader,total = len(train_loader))
        # Reading batches of data
        for b_idx,data in enumerate(tk0):
            for key,value in data.items():
                data[key] = value.to(self.config.device)
            # Zero grading optimizer
            self.optimizer.zero_grad()
            # Output
            output = self.model(data['features'])
            if self.config.loss_params['name']== 'MAE':
                loss   = self.criterion(output,data['pressure'])
            elif self.config.loss_params['name']== 'Huber':
                loss_mask = data['u_out'] == 0
                loss = self.criterion(output[loss_mask], data['pressure'][loss_mask])
            else:
                loss   = self.criterion(output,data['pressure'],data['u_out']).mean()
            # Calculate gradients
            loss.backward()
            # Update Optimizer
            self.optimizer.step()
            # Update Scheduler
            if (self.config.scheduler_params['step_on'] == 'batch')&(self.config.scheduler_params['name'] not in ['Plateu',None]):
                self.scheduler.step()                
            # Saving outputs:
            
            losses.update(loss.detach().item(), train_loader.batch_size)
            tk0.set_postfix(Train_Loss = losses.avg, LR = self.optimizer.param_groups[0]['lr'])
        if (self.config.scheduler_params['step_on']=='epoch')&(self.config.scheduler_params['name'] not in ['Plateu',None]):
                self.scheduler.step()
                
        return losses.avg
    
    def valid_fn(self,valid_loader):
        self.model.eval()
        outputs = []
        targets = []
        # Initialize object Average Meter
        losses = AverageMeter()
        tk0 = tqdm(valid_loader,total = len(valid_loader))
        with torch.no_grad():
            for b_idx,data in enumerate(tk0):
                for key,value in data.items():
                    data[key] = value.to(self.config.device)
                output = self.model(data['features']) 
                
                if self.config.loss_params['name']== 'MAE':
                    loss   = self.criterion(output,data['pressure'])
                    
                elif self.config.loss_params['name']== 'Huber':
                    loss_mask = data['u_out'] == 0
                    loss = self.criterion(output[loss_mask], data['pressure'][loss_mask])
                    
                else:
                    loss   = self.criterion(output,data['pressure'],data['u_out']).mean()
                losses.update(loss.detach().item(), valid_loader.batch_size)
                tk0.set_postfix(Eval_Loss=losses.avg)
                 # Saving outputs:
                outputs.append(output.cpu().detach().numpy())
                targets.append(data['pressure'].cpu().detach().numpy())
        outputs = np.vstack(outputs)
        targets = np.vstack(targets)
        print(outputs.shape,targets.shape)
        return losses.avg,outputs,targets
    
    def fit(self,train = None, valid = None):
        
        seed_everything(self.config.seed)
        train_dataset = VentilatorDataset(train,self.config.cols)
        valid_dataset = VentilatorDataset(valid,self.config.cols)
        #inspiratory_ids = valid[valid['u_out'] == 0].index.values
        
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size  = self.config.batch_size,
                                                   pin_memory  = True,
                                                   num_workers = self.config.num_workers,
                                                   shuffle     = True)
        
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size  = self.config.batch_size,
                                                   num_workers = self.config.num_workers,
                                                   shuffle     = False,
                                                   pin_memory  = True)
        if self.config.previous_path is not None:
            print('Loading Previous Model:',os.path.join(self.config.previous_path,f'fold_{self.config.fold}','model.pt'))
            self.model.load_state_dict(torch.load(os.path.join(self.config.previous_path,f'fold_{self.config.fold}','model.pt')))
            
        self.model.to(self.config.device)
        self.criterion.to(self.config.device)
        es = EarlyStopping (patience = self.config.early_stopping, mode = self.config.mode, delta = 0)
        for epoch in range(self.config.epochs):
            print(f'================= EPOCH: {epoch + 1} =================')
            time.sleep(0.5)
            print("**** Training **** ")
            time.sleep(0.5)
            train_loss = self.train_fn(train_loader)
            print("**** Validation ****")
            time.sleep(0.5)
            valid_loss,valid_outputs,valid_targets = self.valid_fn(valid_loader)
            # Plateu if initialized:
            
            if (self.config.scheduler_params['step_on']=='epoch')&(self.config.scheduler_params['name'] == 'Plateu'):
                self.scheduler.step(valid_metrics[self.config.scheduler_params['step_metric']])
            # Compute Metric
            #score =  mean_absolute_error(valid_targets[inspiratory_ids], valid_outputs[inspiratory_ids])
            valid_metrics = {'valid_loss':valid_loss}
            
            if self.config.logging:
                self.run.log({
                    "train_loss" : train_loss,
                    "valid_loss" : valid_loss,
                    #"score"      : score,
                    "epoch"      : epoch,
                })
                
            os.makedirs(self.config.output_path,exist_ok = True)
            if self.config.save_epoch is not None:
                if ((epoch + 1)%self.config.save_epoch == 0)&(epoch != 0):
                    print(f'Save model at epoch {epoch+1}')
                    torch.save(self.model.state_dict(),os.path.join(self.config.output_path,f'model_epoch_{epoch+1}.pt'))

            es(valid_metrics[self.config.scheduler_params['step_metric']], self.model,os.path.join(self.config.output_path,'model.pt'))
            if es.early_stop:
                print('Meet early stopping')
                self._clean_cache()
                if self.config.logging:
                    self.run.log({'best_score':es.get_best_val_score()})
                    self.run.log({'epoch_es':epoch})
                return es.get_best_val_score()
            
            self._clean_cache()
        print("Didn't meet early stopping")
        if self.config.logging:
            self.run.log({'best_score':es.get_best_val_score()})
            self.run.log({'epoch_es':epoch})
        return es.get_best_val_score()
    
    def predict(self,test = None,path = None):
        assert path != None
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        test_dataset = VentilatorDataset(test,self.config.cols)
        test_loader  = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size  = self.config.batch_size,
                                                  pin_memory  = True,
                                                  num_workers = self.config.num_workers,
                                                  shuffle     = False)
        tk0 = tqdm(test_loader,total = len(test_loader))
        outputs = []
        # Loading weights
        for b_idx,data in enumerate(tk0):
            for key,value in data.items():
                data[key] = value.to(self.config.device)
            # Appending Output and Targets:
            outputs.append(self.model(data['features']).cpu().detach().numpy())
        outputs   = np.vstack(outputs)
        breath_id = test_dataset._get_breathid()
        return breath_id,outputs
        
    def _fetch_loss(self):
        '''
        Add any loss that you want
        '''    
        loss_params = self.config.loss_params
        if loss_params['name'] == 'MAE': return nn.L1Loss()  
        elif loss_params['name'] == 'MAE_FILTERED': return MAE_FILTERED()
        elif loss_params['name'] == 'Huber': return torch.nn.HuberLoss(delta=loss_params['delta'])
        else: raise Exception('Please select a valid loss')
            
        
    def _fetch_scheduler(self):
        '''
        Add any scheduler you want
        '''    
        if self.optimizer is None:
            raise Exception('First choose an optimizer')
        
        else:
            sch_params = self.config.scheduler_params
            
            if sch_params['name'] == 'StepLR':
                return torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                       step_size = sch_parmas['step_size'], 
                                                       gamma     = sch_params.get('gamma',0.1))
            elif sch_params['name'] == 'Plateu': 
                return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                  mode      = self.config.mode, 
                                                                  factor    = sch_params.get('factor',0.1), 
                                                                  patience  = sch_params['patience'], 
                                                                  threshold = 0)
            elif sch_params['name'] == 'CosineAnnealingLR':
                return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                         T_max      = sch_params['T_max'], 
                                         eta_min    = sch_params['min_lr'],
                                         verbose    = True,
                                         last_epoch = -1)
            
            elif sch_params['name'] == 'CosineAnnealingWarmRestarts':
                return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 
                                                                            T_0 = sch_params['T_0'], 
                                                                            T_mult  = 1, 
                                                                            eta_min = sch_params['min_lr'], 
                                                                            last_epoch = -1)
            elif sch_params['name'] == None:
                return None
            else:
                raise Exception('Please choose a valid scheduler')                                       
                
        
    def _fetch_optimizer(self):
        '''
        Add any optimizer you want
        '''
        op_params = self.config.optimizer_params
                                                       
        if op_params['name'] == 'Adam':
            return torch.optim.Adam(self.model.parameters(),lr = self.config.lr, weight_decay = op_params.get('WD',0))
        if op_params['name'] == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr = self.config.lr , weight_decay = op_params.get('WD',0))
        else: 
            raise Exception('Please choose a valid optimizer')
    
    def _clean_cache(self):
        torch.cuda.empty_cache()
        gc.collect()
        
    def _start_group_wandb(self):
        if self.config.logging:
            run = wandb.init(project = self.config.project_name,
                             group   = self.config.experiment_name,
                             config  = get_attributes_config(self.config),
                             save_code = True,
                             reinit    = True)
            run.config.update({"fold": self.config.fold}) # Fold running the model
            run.name = f'fold_{self.config.fold}'
            run.save()
            return run
        else:
            return None
    
    def _log(self,input_dict):
        if self.config.logging:
            return self.run.log(input_dict)
    
    def _upload_df(self,name = None,data = None):
        if self.config.logging:
            table = wandb.Table(dataframe = data)
            self.run.log({name:table})
            
    def _finish(self):
        self.run.finish()
