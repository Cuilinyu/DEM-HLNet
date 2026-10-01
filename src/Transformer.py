import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin,RegressorMixin
import numpy as np

import pandas as pd


import torch
from torch import nn
from torch.utils import data

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0., pred=True):
        super().__init__()
        #out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.q = nn.Linear(in_features, in_features)
        self.k = nn.Linear(in_features, in_features)
        self.v = nn.Linear(in_features, in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, hidden_features)
        self.fc4 = nn.Linear(hidden_features, hidden_features)
        self.fc5 = nn.Linear(hidden_features, hidden_features)
        self.fc6 = nn.Linear(hidden_features, hidden_features)
        self.act = act_layer()
        self.pred = pred
        
        if pred==True:
            self.fcx = nn.Linear(hidden_features,1)
        else:
            self.fcx = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x0 = x
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)
        attn = (q @ k.transpose(-2, -1))
        #print(attn.size())
        attn = attn.softmax(dim=-1)
        x = (attn @ v).squeeze(2)
        #print(x.size())
        x += x0
        x1 = x
        x = self.fc1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        x = self.act(x)
        x = self.fc5(x)
        x = self.act(x)
        x = self.fc6(x)
        x = self.act(x)

        x = self.drop(x)
        x = self.fcx(x)
        x = self.drop(x)
        if self.pred==False:
            x += x1

        x = x.squeeze(0)

        return x
    
class TF(nn.Module):
    def __init__(self, in_features, drop=0.):
        super().__init__()
        self.Block1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_2 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_3 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        self.Block2 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=True)

    def forward(self, x):
        return self.Block2(self.Block1(x))

def get_net(in_features):
    #net = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64,1)).to(device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = TF(in_features=in_features, drop=0.).to(device)
    return net

# 加载数据
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 训练/预测
def train_and_pred(train_features, test_features, train_labels, 
                   num_epochs, learning_rate, weight_decay, batch_size,in_features):
    net = get_net(in_features)
    train_iter = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            loss = nn.MSELoss()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

    preds = net(test_features).cpu().detach().numpy()

    return preds

class transformer(BaseEstimator, RegressorMixin):
    def __init__(self, num_epochs=100,lr=0.01, weight_decay=0.01,batch_size=256 , loss = nn.MSELoss,in_features=10):
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.loss = loss
        self.in_features = in_features

    def fit(self, train_features, train_labels):
        self.train_features=train_features
        self.train_labels=train_labels

        return self

    def predict(self, test_features):
        # 转化为Tensor数据
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_features = torch.tensor(self.train_features, dtype=torch.float32).to(device)
        test_features = torch.tensor(test_features, dtype=torch.float32).to(device)
        train_labels = torch.tensor(self.train_labels.reshape(-1,1), dtype=torch.float32).to(device)
        preds=train_and_pred(train_features, test_features, train_labels, 
                   self.num_epochs, self.lr, self.weight_decay, self.batch_size,self.in_features)


        return preds[:,0]
    
