import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import numpy as np
from utils import *

#NMTucker-L1  our 1 layer NMTucker model
class ML1(nn.Module):
    def __init__(self, data_shape, core_shape):
        super().__init__()
        #Embedding module-initial three embedding matrix U,V,W
        self.U = torch.nn.Embedding(data_shape[0], core_shape[0]) #the mode 1 factor matrix U
        self.V = torch.nn.Embedding(data_shape[1], core_shape[1]) #the mode 2 factor matrix V
        self.W = torch.nn.Embedding(data_shape[2], core_shape[2]) #the mode 3 factor matrix W
        self.G = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, core_shape), dtype=torch.float, requires_grad=True))
        
        xavier_normal_(self.U.weight.data)
        xavier_normal_(self.V.weight.data)
        xavier_normal_(self.W.weight.data)

        self.loss = torch.nn.MSELoss()
        self.core_shape = core_shape

    def forward(self, u_idx, v_idx, w_idx):
        # Embedding model, extract three embeddings for each sample in batch
        r1, r2, r3 = self.core_shape
        u = self.U(u_idx).view(-1, r1) # (B, r1)
        v = self.V(v_idx).view(-1, 1, r2) # (B, 1, r2)
        w = self.W(w_idx).view(-1, 1, r3) # (B, 1, r3)
        
        # Nonlinear Tucker multiplication module
        x = torch.mm(u, self.G.view(r1, -1)) # (B, r1) x (r1, r2*r3) = (B, r2*r3)
        x = torch.sigmoid(x)
        x = x.view(-1, r2, r3) # (B, r2, r3)

        G_mat = torch.bmm(v, x) # (B, 1, r2) bmm (B, r2, r3) = (B, 1, r3)
        G_mat = torch.sigmoid(G_mat)
        G_mat = G_mat.view(-1, r3, 1) # (B, r3, 1)

        x = torch.bmm(w, G_mat) # (B, 1, r3) bmm (B, r3, 1) => (B, 1, 1)
        pred = torch.squeeze(x)
        return pred
        
#NMTucker-L2  our 2 layer NMTucker model
class ML2(torch.nn.Module):
    def __init__(self, data_shape, core_shape, core2_shape):
        super(ML2, self).__init__()
        #Embedding module-initial three embedding matrix U1,V1,W1 in the first layer and the core tensor G1 in the first layer.
        self.U1 = torch.nn.Embedding(core_shape[0], core2_shape[0])
        self.V1 = torch.nn.Embedding(core_shape[1], core2_shape[1])
        self.W1 = torch.nn.Embedding(core_shape[2], core2_shape[2])
        self.G1 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (core2_shape[0], core2_shape[1], core2_shape[2])), 
                                     dtype=torch.float,
                                     device="cuda",
                                     requires_grad=True))
        
        #Embedding moduke-initial three embedding matrix U,V,W in the second layer (output layer).
        self.U = torch.nn.Embedding(data_shape[0], core_shape[0])
        self.V = torch.nn.Embedding(data_shape[1], core_shape[1])
        self.W = torch.nn.Embedding(data_shape[2], core_shape[2])

        self.loss = torch.nn.MSELoss()
        self.core_shape = core_shape

        xavier_normal_(self.U1.weight.data)
        xavier_normal_(self.V1.weight.data)
        xavier_normal_(self.W1.weight.data)
        xavier_normal_(self.U.weight.data)
        xavier_normal_(self.V.weight.data)
        xavier_normal_(self.W.weight.data)
   
    def forward(self, u_idx, v_idx, w_idx):
        r1, r2, r3 = self.core_shape
        #Embedding model -extract three embedding vectors (for here is three batches of vectors)
        u = self.U(u_idx) 
        u = u.view(-1, r1) # (B, r1)
        v = self.V(v_idx)
        v = v.view(-1, 1, r2) # (B, 1, r2)
        w = self.W(w_idx)
        w = w.view(-1, 1, r3) # (B, 1, r3)
        
        #reconstruct the core tensor G in the output layer (for more information see the section "Implicit Regularization with Multiple LayerRecursive NMTucker")
        res = mode_dot(self.G1, self.U1.weight, 0)
        res = torch.sigmoid(res)
        res = mode_dot(res, self.V1.weight, 1)
        res = torch.sigmoid(res)
        G = mode_dot(res, self.W1.weight, 2) #the reconstructed core tensor G in the output layer
        G = torch.sigmoid(G)
        
        #Nonlinear Tucker multiplication module
        x = torch.mm(u, G.reshape(r1, -1)) # (B, r1) mm (r1, r2*r3) = (B, r2*r3)
        x = torch.sigmoid(x)
        x = x.view(-1, r2, r3) # (B, r2, r3)
   
        G_mat = torch.bmm(v, x) # (B, 1, r2) bmm (B, r2, r3) = (B, 1, r3)
        G_mat = torch.sigmoid(G_mat)
        G_mat = G_mat.view(-1, r3, 1) # (B, r3, 1)

        x = torch.bmm(w, G_mat) # (B, 1, r3) bmm (B, d3, 1) = (B, 1, 1)
        pred = torch.squeeze(x)
        return pred
    
#NMTucker-L3  our 3 layer NMTucker model
class ML3(torch.nn.Module):
    def __init__(self, data_shape, core_shape, core2_shape, core3_shape):
        super(ML3, self).__init__()
        
        #Embedding module-initial three embedding matrix U1,V1,W1 in the first layer and the core tensor G1 in the first layer.
        self.U1 = torch.nn.Embedding(core2_shape[0], core3_shape[0])
        self.V1 = torch.nn.Embedding(core2_shape[1], core3_shape[1])
        self.W1 = torch.nn.Embedding(core2_shape[2], core3_shape[2])
        self.G1 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (core3_shape[0], core3_shape[1], core3_shape[2])), 
                                     dtype=torch.float,
                                     device="cuda", requires_grad=True))
        
        #Embedding module-initial three embedding matrix U2,V2,W2 in the second layer.
        self.U2 = torch.nn.Embedding(core_shape[0], core2_shape[0])
        self.V2 = torch.nn.Embedding(core_shape[1], core2_shape[1])
        self.W2 = torch.nn.Embedding(core_shape[2], core2_shape[2])
        
        #Embedding moduke-initial three embedding matrix U,V,W in the third layer (output layer).
        self.U = torch.nn.Embedding(data_shape[0], core_shape[0])
        self.V = torch.nn.Embedding(data_shape[1], core_shape[1])
        self.W = torch.nn.Embedding(data_shape[2], core_shape[2])

        self.loss = torch.nn.MSELoss()
        self.core_shape = core_shape
        self.core2_shape = core2_shape
       
        xavier_normal_(self.U1.weight.data)
        xavier_normal_(self.V1.weight.data)
        xavier_normal_(self.W1.weight.data)
        xavier_normal_(self.U2.weight.data)
        xavier_normal_(self.V2.weight.data)
        xavier_normal_(self.W2.weight.data)
        xavier_normal_(self.U.weight.data)
        xavier_normal_(self.V.weight.data)
        xavier_normal_(self.W.weight.data)
   
    def forward(self, u_idx, v_idx, w_idx):
        r1, r2, r3 = self.core_shape
        #Embedding model -extract three embedding vectors (for here is three batches of vectors)
        u = self.U(u_idx) 
        u = u.view(-1, r1) # (B, r1)
        v = self.V(v_idx)
        v = v.view(-1, 1, r2) # (B, 1, r2)
        w = self.W(w_idx)
        w = w.view(-1, 1, r3) # (B, 1, r3)
           
        #reconstruct the core tensor G2 in the second layer (for more information see the section "Implicit Regularization with Multiple LayerRecursive NMTucker")
        G2=torch.tensor(multi_mode_dot(self.G1,
                                       [self.U1(torch.tensor(range(self.core2_shape[0])).cuda()),
                                        self.V1(torch.tensor(range(self.core2_shape[1])).cuda()),
                                        self.W1(torch.tensor(range(self.core2_shape[2])).cuda())],
                                       modes=[0,1,2]))
    
        #reconstruct the core tensor G in the output layer
        G=torch.tensor(multi_mode_dot(G2,
                                     [self.U2(torch.tensor(range(r1)).cuda()),
                                      self.V2(torch.tensor(range(r2)).cuda()),
                                      self.W2(torch.tensor(range(r3)).cuda())],
                                      modes=[0,1,2]))
                                  
        #Nonlinear Tucker multiplication module
        x = torch.mm(u, G.contiguous().view(r1, -1)) # (B, r1) mm (r1, r2*r3) = (B, r2*r3)
        x = torch.nn.functional.sigmoid(x) #first non-linear activation function
        x = x.view(-1, r2, r3) # (B, r2, r3)

        G_mat = torch.bmm(v,x) #(B, 1, r2) bmm (B, r2, r3) = (B,1,r3)
        G_mat = torch.nn.functional.sigmoid(G_mat) #second non-linear activation function
        G_mat = G_mat.view(-1, r3, 1) #(B, r3, 1)

        x = torch.bmm(w, G_mat) #(B,1,r3) bmm (B, r3, 1) = (B, 1, 1)
        pred = torch.squeeze(x)
        return pred