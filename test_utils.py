import numpy as np
import torch as t
import torchvision as tv
import torchvision.transforms as transform
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

def get_lr(optimizer):#directly change learning rate of the optimizer
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def change_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr'] = new_lr

def imshow(img):
    img = img /2. + .5  #convert to 0 - 0.1 as PIL image?
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def eval_accuracy(basenet, topnet, testloader, labels, device):
    correct = 0
    total = 0
    with t.no_grad():#won't add any new tensor into the graph
        for data in testloader:
            images, labels = data
            images = images.type(t.FloatTensor).to(device)
            labels = labels.to(device)
            
            fx = basenet(images)
            if topnet != None:
                fx = topnet(fx)
            
            _, predicted = t.max(fx.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total