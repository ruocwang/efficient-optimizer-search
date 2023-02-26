import glob
from urllib.request import HTTPDigestAuthHandler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
import torchvision
from nos.meta_module import MetaModule, MetaLinear


USE_CUDA = torch.cuda.is_available()

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v


class MNISTLoss:
    def __init__(self, training=True):
        dataset = datasets.MNIST(
            '/home/chenwy/mnist', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        indices = list(range(len(dataset)))
        np.random.RandomState(10).shuffle(indices)
        if training:
            indices = indices[:len(indices) // 2]
        else:
            indices = indices[len(indices) // 2:]

        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=128,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

        self.batches = []
        self.cur_batch = 0
        
    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch


class MNISTNet(MetaModule):
    def __init__(self, hidden_dim=20, n_layers=1, **kwargs):
        super().__init__()

        inp_size = 28*28
        self.layers = {}
        for i in range(n_layers):
            self.layers[f'mat_{i}'] = MetaLinear(inp_size, hidden_dim)
            inp_size = hidden_dim

        self.layers['final_mat'] = MetaLinear(inp_size, 10)
        self.layers = nn.ModuleDict(self.layers)

        self.activation = nn.Sigmoid()
        self.loss = nn.NLLLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]
    
    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 28*28)))
        out = w(Variable(out))

        cur_layer = 0
        while f'mat_{cur_layer}' in self.layers:
            inp = self.activation(self.layers[f'mat_{cur_layer}'](inp))
            cur_layer += 1

        inp = F.log_softmax(self.layers['final_mat'](inp), dim=1)
        l = self.loss(inp, out)
        return l
    

class MNISTNetV2(nn.Module):
    def __init__(self, hidden_dim=20, n_layers=1, **kwargs):
        super().__init__()
        input_size = 28 * 28
        n_classes = 10

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_dim))
        for i in range(1, n_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.final_layer = nn.Linear(hidden_dim, n_classes)

        self.activation = nn.Sigmoid()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]
    
    def forward(self, x): # x = (b, c, h, w)
        x = x.view(x.shape[0], -1)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        
        x = F.log_softmax(self.final_layer(x), dim=1)
        return x
        
        # inp, out = loss.sample()
        # inp = w(Variable(inp.view(inp.size()[0], 28*28)))
        # out = w(Variable(out))

        # cur_layer = 0
        # while f'mat_{cur_layer}' in self.layers:
        #     inp = self.activation(self.layers[f'mat_{cur_layer}'](inp))
        #     cur_layer += 1

        # inp = F.log_softmax(self.layers['final_mat'](inp), dim=1)
        # l = self.loss(inp, out)
        # return l
        

class MNISTNetV2_2Layer(MNISTNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(n_layers=2, *args, **kwargs)
        
class MNISTNetV2_Big(MNISTNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(hidden_dim=40, *args, **kwargs)
        
class MNISTNetV2_Relu(MNISTNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = nn.ReLU()