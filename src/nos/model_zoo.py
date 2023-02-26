import torch
import os
import torchvision

from nos.resnet import ResNet18
from nos.lenet import LeNet
from nos.mnistnet import MNISTNet, MNISTNetV2, MNISTNetV2_2Layer, MNISTNetV2_Big, MNISTNetV2_Relu
from nos.linearnet import LinearNet
from nos.conv import Conv, ConvQuoc, ConvDefault

from utils.logging import log_and_print


def load_model(name):
    adv_ckpt_path = '/nfs/data/ruocwang/projects/crash/automl/nos_tasks/BayesianDefense/checkpoint/'
    device = torch.cuda.current_device()
    
    #### clfs/regression
    if name == 'LeNet':
        model = LeNet()
    elif name == 'ResNet18':
        model = ResNet18()
    elif name == 'MNISTNet': # learning2learn with gd with gd paper
        model = MNISTNet()
    elif name == 'MNISTNetV2': # learning2learn with gd with gd paper
        model = MNISTNetV2()
    elif name == 'MNISTNetV2_2Layer': # learning2learn with gd with gd paper
        model = MNISTNetV2_2Layer()
    elif name == 'MNISTNetV2_Big': # learning2learn with gd with gd paper
        model = MNISTNetV2_Big()
    elif name == 'MNISTNetV2_Relu': # learning2learn with gd with gd paper
        model = MNISTNetV2_Relu()
    elif name == 'Conv': # Quoc
        model = Conv()
    elif name == 'ConvQuoc': # Quoc
        model = ConvQuoc()
    elif name == 'ConvDefault':
        model = ConvDefault()
    elif name == 'LinearNet':
        model = LinearNet(bias=False)
    
    ## BayesianDefense
    elif name == 'VGG-plain':
        from adv_models.vgg import VGG
        try:
            model = VGG('VGG16', 10, img_width=32)
            model.load_state_dict(torch.load(os.path.join(adv_ckpt_path, 'cifar10_vgg_plain_single.pth')))
        except:
            print('failed to load single-cpu ckpts, making one now')
            model = torch.nn.DataParallel(VGG('VGG16', 10, img_width=32), device_ids=[0])
            model.load_state_dict(torch.load(os.path.join(adv_ckpt_path, 'cifar10_vgg_plain.pth')))
            model.to('cpu')
            torch.save(model.module.state_dict(), os.path.join(adv_ckpt_path, 'cifar10_vgg_plain_single.pth'))
            exit()
    elif name == 'VGG-adv':
        from adv_models.vgg import VGG
        try:
            model = VGG('VGG16', 10, img_width=32)
            model.load_state_dict(torch.load(os.path.join(adv_ckpt_path, 'cifar10_vgg_adv_single.pth')))
        except:
            print('failed to load single-cpu ckpts, making one now')
            model = torch.nn.DataParallel(VGG('VGG16', 10, img_width=32), device_ids=[0])
            model.load_state_dict(torch.load(os.path.join(adv_ckpt_path, 'cifar10_vgg_adv.pth')))
            model.to('cpu')
            torch.save(model.module.state_dict(), os.path.join(adv_ckpt_path, 'cifar10_vgg_adv_single.pth'))
            exit()

    ## imagenet models
    elif name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'vit_small_patch16_224':
        from timm.models import create_model
        from foolbox import PyTorchModel
        model = create_model(name, pretrained=True, num_classes=1000, in_chans=3,)
        model = torch.nn.DataParallel(model, device_ids=[device])
        model.eval()
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3) ## preprocessing outside
        model = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing, device=device)
    
    ## robust bench models
    else:
        from foolbox import PyTorchModel
        from robustbench.utils import load_model as load_model_rb
        model = load_model_rb(model_name=name, dataset='cifar10', threat_model='Linf')
        assert not model.training
        model = PyTorchModel(model, bounds=(0, 1), device=device)

    return model