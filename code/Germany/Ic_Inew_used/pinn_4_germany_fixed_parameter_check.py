# -*- coding: utf-8 -*-

import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# 将上两级目录加入到系统路径中（constants.py所在目录）
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import constants
import numpy as np
import pandas as pd

# loading dataset
path, paras = constants.read_data('Germany')
pf = pd.read_csv(path)

country = paras[0]
N = int(paras[1])
days =int(paras[-1]) 
date = np.array(pf['date'])

# data over N
Ic_raw = np.array(pf['Ic'])/N
Inew_raw = np.array(pf['Inew'])/N
Recovered_raw = np.array(pf['R'])/N
D_raw = np.array(pf['D'])/N
N = N/N

top = int(np.argmax(Inew_raw)) 

# 训练集/测试集
train_size = top-5
test_size = -days

device = constants.get_device()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.001)


class Net_SIR(nn.Module):
    def __init__(self):
        super(Net_SIR, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Linear(1,32),nn.Tanh(), # 1 means time
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,1),nn.Sigmoid() 
        )

    def forward(self, x):
        output = self.layers(x)
        Ic = output[:,0]
        Inew = Ic[1:]- Ic[:-1]
        return Ic,Inew

    def set_layer(self, layers):
        self.layers = layers


# 初始化 NNs
pinn_sir = Net_SIR()
pinn_sir.apply(init_weights)
pinn_sir = pinn_sir.to(device)


def network(t):
    input = t.reshape(-1,1)
    input = torch.from_numpy(input).float().to(device)
    return pinn_sir(input)


def data_loss(Inew,index):

    loss_inew = (Inew[index] - torch.from_numpy(Inew_raw[index]).to(device))**2
    return loss_inew.sum()

optimizer = optim.Adam(pinn_sir.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.998)

# 训练集离散时间点
t_train = np.linspace(0,train_size,train_size+1)[:-1]
# 训练集时间点shuffle
index = torch.randperm(train_size-1)

total_epoch_loss,total_data_loss,total_residuals_loss = [],[],[]
early_stopping = 2000
#双头队列
loss_history = deque(maxlen=early_stopping+ 1)

pinn_sir.zero_grad()

for epoch in range(1000000):  # loop over the dataset multiple times
    running_loss = 0.0        
    # zero the parameter gradients
    optimizer.zero_grad()

    Ic,Inew = network(t_train)
    loss_inew = data_loss(Inew,index)

    loss = 10000*500*3*loss_inew

    running_loss += loss.item()
    loss_history.append(running_loss)

    total_epoch_loss.append(running_loss)

    # 如果队列满了并且弹出的第一个值小于队列中剩余的最小值，表明该值为最小值，在队列长度个epoch内loss不再下降
    if len(loss_history) > early_stopping and loss_history.popleft() < min(loss_history):
      print(f"Early stopping at [{epoch}] times. No train loss improvement in [{early_stopping}] epochs.")
      break
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if (epoch % 1000 == 0):   
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch = {epoch}, loss: {running_loss}, lr: {lr}.')
        print(f'data loss: Inew loss: {loss_inew.item()}.')
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

print(f'Finished Training.')


# 绘制loss图
constants.plot_loss(country,total_epoch_loss,total_epoch_loss,total_epoch_loss,'fixed')

t = np.linspace(0,days,days+1)[:-1]

Ict,Inewt = network(t)
inewt = Inewt.cpu().detach().numpy()

# Ic,Inew
constants.plot_results_comparation(country,'Inew',Inew_raw[:-1], inewt, inewt, train_size,'fixed')

# 保存结果
# constants.save_results_fixed_parameters(country,date,st,it,rt,ict,np.append(inewt,inewt[-1]),Ic_raw,Inew_raw[:-1],S_ode,I_ode,R_ode,Ic_ode)

# 保存parameters learned. [S0,I0,R0,beta,gamma]
# constants.save_fixed_parameters_result(country,beta_init,gamma_init,S0_init,I0_init,R0_init,beta_final,gamma_final,S0_final,I0_final,R0_final)


# 打印beta 和 gamma
print(f'peak index is: {top}.')
# print(f'beta and gamma is: {beta_final}, {gamma_final}, beta/gamma: {beta_final/gamma_final}, S0_final: {S0_final}, I0_final: {I0_final}, R0_final: {R0_final}.')