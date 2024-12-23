import argparse
import torch
import random
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from model import Model
from utils import *

import numpy as np
import torch.nn.functional as F

def main(args):
    print('Test diverse distributions of Cifar-100')
    
    model = Model(backbone='resnet32', 
                  num_classes=args.num_classes, 
                  num_experts=args.num_experts,
                  reduce_dimension=False,
                  use_norm=False,
                  share_layer3=False)
    
    print('Loading checkpoint: {} ...'.format(args.resume))
    checkpoint = torch.load(args.resume)

    state_dict = checkpoint['state_dict']
    
    model.load_state_dict(state_dict)
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    num_classes = args.num_classes
    
    distrb = {
        'uniform': (0,False),
        'forward50': (0.02, False),
        'forward25': (0.04, False), 
        'forward10':(0.1, False),
        'forward5': (0.2, False),
        'forward2': (0.5, False),
        'backward50': (0.02, True),
        'backward25': (0.04, True),
        'backward10': (0.1, True),
        'backward5': (0.2, True),
        'backward2': (0.5, True),
    }  
    
    record_list=[]
    
    test_distribution_set = ["uniform",  "forward50",  "forward25", "forward10", "forward5", "forward2", "backward2", "backward5", "backward10", "backward25", "backward50"] 
    for test_distribution in test_distribution_set: 
        print('*', test_distribution)
        data_loader = TestAgnosticImbalanceCIFAR100DataLoader(
            data_dir='../datasets',
            batch_size=128,
            shuffle=False,
            training=False,
            num_workers=1,
            test_imb_factor=distrb[test_distribution][0],
            reverse=distrb[test_distribution][1]
        )
        record = validation(data_loader, model, num_classes,device, args)
            
        record_list.append(record)
    print('='*25, ' Final results ', '='*25)
    i = 0
    for txt in record_list:
        print(test_distribution_set[i]+'\t')
        print(*txt)          
        i+=1
   

def validation(data_loader, model, num_classes,device,  args):
    
    top1 = AverageMeter('Acc@1', ':6.2f')
    head = AverageMeter('Head', ':6.2f')
    medium = AverageMeter('Medium', ':6.2f')
    tail = AverageMeter('Tail', ':6.2f')  
    
    with torch.no_grad():
        for i, (data, indice) in enumerate(tqdm(data_loader)):
            # Sort categories by frequency and assign new category labels
            target = torch.tensor([args.class_mapping[i] for i in indice])
            
            data, target = data.to(device), target.to(device)
            logits, w = model(data, target, args, 'test')
            if args.pa:
                logits = [w[:, i] * logits[i] for i in range(args.num_experts)]
            
            logit_h, logit_m, logit_t = logits
            output = torch.stack([logit_h, logit_m, logit_t], dim=1).sum(1)
            output = F.softmax(output, dim=1)
            
            acc1, acc5 = Accuracy(output, target, topk=(1, 5))
            head_acc, head_sample_num, mid_acc, mid_sample_num, tail_acc, tail_sample_num = Accuracy_Diffshot(output, target, args)
            top1.update(acc1[0], data.size(0))
    
            head.update(head_acc[0], head_sample_num)      
            medium.update(mid_acc[0], mid_sample_num)    
            tail.update(tail_acc[0], tail_sample_num) 
                     
    print('Val_Top1:{:.3f} Head:{:.3f}  Medium:{:.3f}  Tail:{:.3f}'.format(top1.avg,head.avg, medium.avg, tail.avg))
    
    return top1.avg.cpu().detach().numpy(), head.avg.cpu().detach().numpy(), medium.avg.cpu().detach().numpy(), tail.avg.cpu().detach().numpy()

    
class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size
    
    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets]) # AcruQRally we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num # Ensures every instance has the chance to be visited in an epoch


class  TestAgnosticImbalanceCIFAR100DataLoader(DataLoader):
    """
    Imbalance Cifar100 Data Loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=False, balanced=False, 
                 retain_epoch_size=True, 
                 imb_type='exp', imb_factor=0.01, test_imb_factor=0, reverse=False):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ])
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_trsfm) # test set
        

        if test_imb_factor!=0:
            dataset = IMBALANCECIFAR100(data_dir, train=False, download=True, transform=train_trsfm, imb_type=imb_type, imb_factor=test_imb_factor, reverse=reverse)
        else:
            dataset = test_dataset
        val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 100

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None
        
        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler) # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default='log/cifar100_resnet32_3_0.01_4/checkpoint.best.pth.tar', type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('--num_experts', default=3)
    parser.add_argument('--num_classes', default=100)  
    parser.add_argument('--pa', default=True)  
    
    args = parser.parse_args()

    train_data = IMBALANCECIFAR100(root='../datasets', imb_type='exp', imb_factor=0.01, rand_number=42, train=True, download=True, transform=None)
    cls_num_list = train_data.get_cls_num_list()
    cls_num_sorted = np.argsort(-np.array(cls_num_list))
    args.class_mapping = [0 for i in range(len(cls_num_sorted))]
    for i in range(len(cls_num_sorted)):
        args.class_mapping[cls_num_sorted[i]] = i    
        
    args.head_ave, args.med_ave = 35, 70    
    
    main(args)
