import math
import torch
import shutil
import os
import random
import numpy as np
import matplotlib
import torchvision.transforms as transforms
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim import lr_scheduler
from PIL import Image, ImageEnhance, ImageOps

use_fp16 = False

if use_fp16:
    from torch.cuda.amp import autocast
else:
    class Autocast(): # This is a dummy autocast class
        def __init__(self):
            pass
        def __enter__(self, *args, **kwargs):
            pass
        def __call__(self, arg=None):
            if arg is None:
                return self
            return arg
        def __exit__(self, *args, **kwargs):
            pass

    autocast = Autocast()
    
def prepare_folders(args):
    
    folders_util = [args.root_log, os.path.join(args.root_log, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(args, state, is_best):
    
    filename = 'log/{}/checkpoint.pth.tar'.format(args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def Accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def Accuracy_Diffshot(output, target, args):
    
    with torch.no_grad():

        batch_size = target.size(0)
    
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    
        # create three variables to store the accuracy and sample number for each shot type
        many_acc, mid_acc, few_acc  = 0., 0., 0.
        many_sample_num,  mid_sample_num, few_sample_num = 0, 0, 0
    
        for idx, label in enumerate(target):
            # calculate the accuracy and sample number for each k and add to the corresponding variable
            correct_k = correct[0:, idx][: 1].reshape(-1).float().sum(0, keepdim=True)
            
            if label < args.head_ave:
                many_acc += correct_k
                many_sample_num += 1
                
            elif label >= args.med_ave:
                few_acc += correct_k
                few_sample_num += 1
                
            else:
                mid_acc += correct_k
                mid_sample_num += 1

        assert (many_sample_num + mid_sample_num + few_sample_num) == batch_size
        
        many_acc = many_acc.mul_(100.0 / many_sample_num) if many_sample_num else torch.tensor([0.]).cuda()
        mid_acc = mid_acc.mul_(100.0 / mid_sample_num) if mid_sample_num!=0 else torch.tensor([0.]).cuda()
        few_acc = few_acc.mul_(100.0 / few_sample_num) if few_sample_num!=0 else torch.tensor([0.]).cuda()

        return many_acc.cuda(), many_sample_num, mid_acc.cuda(), mid_sample_num, few_acc.cuda(), few_sample_num

    
def load_pretrained_weights(model, pretrained_path):
    # model: the pytorch model to be loaded
    # pretrained_path: the path of the pretrained pth file
    # load the pretrained weights
    pretrained_dict = torch.load(pretrained_path)
    # get the model state dict
    model_dict = model.state_dict()
    # create a new dict to store the updated weights
    new_dict = {}
    # iterate over the model parameters
    for k, v in model_dict.items():
        # if the parameter name starts with "backbone.", remove "backbone."
        if k.startswith("backbone."):
            k = k[len("backbone."):]
            # if the parameter name is in the pretrained dict, load the pretrained value
            if (k in pretrained_dict) and ('fc' not in k):
                new_dict["backbone." + k] = pretrained_dict[k]
                
            # otherwise, keep the model value
            else:
                new_dict["backbone." + k] = v
        else:
            new_dict[k] = v
    # update the model state dict
    model.load_state_dict(new_dict)
    # return the model
    return model

def load_state_dict(model, state_dict, no_ignore=False):
    own_state = model.state_dict()
    count = 0
    for name, param in state_dict.items():
        if name not in own_state: # ignore
            print("Warning: {} ignored because it does not exist in state_dict".format(name))
            assert not no_ignore, "Ignoring param that does not exist in model's own state dict is not allowed."
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except RuntimeError as e:
            print("Error in copying parameter {}, source shape: {}, destination shape: {}".format(name, param.shape, own_state[name].shape))
            raise e
        count += 1
    if count != len(own_state):
        print("Warning: Model has {} parameters, copied {} from state dict".format(len(own_state), count))
    return count

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.autograd.Variable(x, requires_grad=requires_grad)

def init_lrSchedualer(epoch, optimizer, args):
    lr = 0.
    scheduler_sequence = []

    base_scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epochs, args.lr_decay)  
    if args.cosine_anneal:
        base_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scheduler_sequence.append(base_scheduler)

    if args.warmup:
        warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=args.warmup_factor, end_factor=1., total_iters=args.warmup_epochs)
        scheduler_sequence.append(warmup_scheduler)

    scheduler = lr_scheduler.ChainedScheduler(scheduler_sequence)
    
    lr = warmup_scheduler.get_last_lr() if ((epoch+1) <= args.warmup_epochs) and args.warmup else base_scheduler.get_last_lr()
    
    return scheduler, lr


def adjust_learning_rate(optimizer, epoch, args):

    lr = args.lr

    if epoch <= args.warmup_epochs and args.warmup:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        if args.cosine_anneal:
            eta_min = 0.
            T_max = args.epochs
            lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
        else:            
            for e in args.lr_decay_epochs:
                if epoch == e:
                    args.lr *= args.lr_decay
                    lr = args.lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr