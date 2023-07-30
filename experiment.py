import torch
import numpy as np
from pytorchtools import EarlyStopping
import time
from utils import *
from model import *

class Experiment:
    
    def __init__(self,
                 model='ML1',
                 learning_rate=0.0005,
                 data_shape=(3,4,5,6),
                 core_shape=(2,3,4),
                 core2_shape=(2,3,4),
                 core3_shape=(2,3,4),
                 val_ratio=0.1,
                 num_iterations=500,
                 device='cuda', 
                 batch_size=128,
                 patience=10,
                 tr_idxs=1,
                 tr_vals=1,
                 lambda_l1=1e-3,
                 lambda_l2=1e-3,
                 regularization='L1'):
        
        self.model = model
        self.learning_rate = learning_rate
        self.data_shape = data_shape
        self.core_shape = core_shape
        self.core2_shape = core2_shape
        self.core3_shape = core3_shape
        self.num_iterations = num_iterations
        self.patience = patience
        self.device = device
        self.batch_size = batch_size
        self.tr_idxs = tr_idxs
        self.tr_vals = tr_vals
        self.val_ratio = val_ratio
        self.regularization = regularization
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
                     
    def evaluate(self, model, idxs, vals):
        with torch.no_grad():
            model.eval()
            prediction_list = []
            target_list = []
            losses = []
            rmse_list = []
            mae_list = []
                    
            batch_start = 0
            while batch_start < len(idxs):
                batch_end = batch_start + self.batch_size
                idx_batch = idxs[batch_start: batch_end, :3].reshape(-1, 3) # in case there is 1 sample in batch
                target_batch = vals[batch_start: batch_end]
                
                u_idx = torch.LongTensor(idx_batch[:,0]).to(self.device)
                v_idx = torch.LongTensor(idx_batch[:,1]).to(self.device)
                w_idx = torch.LongTensor(idx_batch[:,2]).to(self.device)
                target_batch = torch.FloatTensor(target_batch).to(self.device)

                predictions = model.forward(u_idx, v_idx, w_idx) 
                prediction_list.extend(predictions.cpu().tolist())
                target_list.extend(target_batch.cpu().tolist())

                loss = model.loss(predictions, target_batch)
                losses.append(loss.item())

                batch_start += self.batch_size
                        
            val_loss = np.mean(losses)  
            val_mae = mae(target_list, prediction_list)
            val_rmse = rmse(target_list, prediction_list)
            val_mape = mape(target_list, prediction_list)

            return prediction_list, val_loss, val_rmse, val_mae, val_mape
    
    def train_and_eval(self):
        train_losses=[]
        val_losses=[]
        val_rmses=[]
        val_maes=[]
        val_mapes=[]
        
        print("Training NMTucker model...")
        train_data, val_data = train_val_split(self.tr_idxs, self.tr_vals, self.val_ratio)
        
        if self.model=='ML1': #NMTucker-L1 
            model = ML1(self.data_shape, self.core_shape)
        elif self.model=="ML2": #NMTucker-L2
            model = ML2(self.data_shape, self.core_shape, self.core2_shape)
        elif self.model=="ML3": #NMTucker-L3
            model = ML3(self.data_shape, self.core_shape, self.core2_shape, self.core3_shape)
             
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        print("Starting training...")
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(train_data)
            
            batch_start = 0
            while batch_start < len(train_data):
                optimizer.zero_grad()
                batch_end = batch_start + self.batch_size
                idx_batch = train_data[batch_start: batch_end, :3].reshape(-1, 3) # in case there is 1 sample in batch
                target_batch = train_data[batch_start: batch_end, -1]

                u_idx = torch.LongTensor(idx_batch[:,0]).to(self.device)
                v_idx = torch.LongTensor(idx_batch[:,1]).to(self.device)
                w_idx = torch.LongTensor(idx_batch[:,2]).to(self.device)
                target_batch = torch.FloatTensor(target_batch).to(self.device)
                    
                predictions = model.forward(u_idx, v_idx, w_idx)  
                loss = model.loss(predictions, target_batch)
                losses.append(loss.item())
                
                if self.regularization=='L1':
                    lossl1 = l1_regularizer(model, self.lambda_l1)
                    loss = loss + lossl1
                elif self.regularization=='L2':
                    lossl2 = l2_regularizer(model, self.lambda_l2)
                    loss = loss + lossl2
                    
                loss.backward(retain_graph=True)
                optimizer.step()
                batch_start += self.batch_size
                
            train_loss = np.mean(losses)
            train_losses.append(train_loss)
    
            with torch.no_grad():
                _, val_loss, val_rmse, val_mae, val_mape = self.evaluate(model, val_data[:,0:3], val_data[:,-1])
                val_losses.append(val_loss)
                val_rmses.append(val_rmse)
                val_maes.append(val_mae)
                val_mapes.append(val_mape)

            #print('TIME ELAPSED:{:.4f}'.format(time.time()-start_train))    
            print(f'EPOCH {it}, TRAIN LOSS:{train_loss:.7f}, VALID LOSS:{val_loss:.7f}')       
            print(f'rmse: {val_rmse:.4f} -- mae: {val_mae:.4f} -- mape: {val_mape:.4f}')

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            print('\n')
                    
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))
                    
        dic = dict()
        dic['train_loss'] = train_losses
        dic['val_loss'] = val_losses
        dic['val_rmse'] = val_rmses
        dic['val_mae'] = val_maes
        dic['val_mape'] = val_mapes
        dic['model'] = model
        
        return dic