import torch

#优化器
def SGD(net,lr=0.01):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    return optimizer

def adam(net,lr=0.01,betas=()):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
    return optimizer

def RMSprop(net,lr=0.01,weight_decay=1e-8, momentum=0.9):
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    return optimizer

#学习率优化算法
def ExponentialLR(optimizer,gamma=0.5):
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma)
    return scheduler