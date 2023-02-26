import copy
import torch
from torch.utils.data import DataLoader, Dataset, sampler, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from torchvision.datasets import mnist, MNIST, ImageFolder



class RandomQuadratic(Dataset):
    def __init__(self, n=10):
        """
        Args:
            //transform (callable, optional): Optional transform to be applied
        """
        self.n = n
        self.resample()

    def resample(self):
        n = self.n
        self.X = torch.randn(n, n)
        self.y = torch.randn(n)
        

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def seed_everything(seed):
    # np.random.seed(seed)
    torch.manual_seed(seed)


def get_val_loader(batch_size, model_name, input_size=224, normalize=False):
    data_dir = "/nfs/data/ruocwang/data/imagenet"

    if '384' in model_name:
        input_size = 384

    if normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        valdir = os.path.join(data_dir, 'val')
        dataset = ImageFolder(valdir, transforms.Compose([
                transforms.Resize(input_size + 32),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]))
        val_loader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=2, shuffle=True, pin_memory=True)
        #### wrc modified
        ## shuffle was True
        ####
    else:
        valdir = os.path.join(data_dir, 'val')
        dataset = ImageFolder(valdir, transforms.Compose([
                transforms.Resize(input_size + 32),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
            ]))
        val_loader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=2, shuffle=True, pin_memory=True)
        #### wrc modified
        ## shuffle was True
        ####

    return val_loader


def load_dataset(name, batch_size,
                 normalize_data=False, model=None): ## 
    
    if name == 'RandomQuadratic':
        n = 10
        assert batch_size == n
        trainset = RandomQuadratic(n=10)
        testset = RandomQuadratic(n=10)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader  = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    elif name == "MNIST":
        data_path = '/nfs/data/ruocwang/data/'
        dataset = MNIST(data_path, train=True, download=False, transform=transforms.ToTensor())
        indices = list(range(len(dataset)))
        np.random.RandomState(10).shuffle(indices)
        
        train_indices = indices[:len(indices) // 2]
        test_indices  = indices[len(indices) // 2:]
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(train_indices),
                                  num_workers=8)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(test_indices),
                                 num_workers=8)
        #### cross validation
        cv_indices = copy.deepcopy(train_indices)
        np.random.RandomState(10).shuffle(cv_indices)
        search_indices = cv_indices[:int(len(cv_indices)*0.8)]
        valid_indices  = cv_indices[:int(len(cv_indices)*0.2)]
        search_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(search_indices),
                                num_workers=4)
        valid_loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(valid_indices),
                                num_workers=4)

    elif name == 'cifar10':
        data_path = '/nfs/data/ruocwang/data/cifar.python/'
        trans = [transforms.ToTensor()]
        if normalize_data:
            trans.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        transform_train = transforms.Compose(trans)
        transform_test = transforms.Compose(trans)
        
        trainset  = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transform_test)

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        #### cross validation
        ratio = 0.9 ## TODO under dev ##
        indices = list(range(len(trainset)))
        np.random.RandomState(10).shuffle(indices)
        search_indices = indices[:int(len(indices)*ratio)]
        valid_indices = indices[int(len(indices)*ratio):]
        
        search_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(search_indices),
                                   num_workers=2)
        valid_loader  = DataLoader(trainset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(valid_indices),
                                   num_workers=2)

    elif name == 'imagenet':
        train_loader  = None
        valid_loader  = get_val_loader(batch_size, model, normalize=normalize_data)
        test_loader   = copy.deepcopy(valid_loader)
        search_loader = copy.deepcopy(valid_loader)

    else:
        print(f'ERROR: UNSUPPORTED DATASET: {name}'); exit(1)

    return train_loader, test_loader, search_loader, valid_loader
