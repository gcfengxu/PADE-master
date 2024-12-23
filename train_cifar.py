import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import argparse
import time
import warnings
import numpy as np
import torch
import imbalance_cifar
import torch.nn.functional as F
import torch.backends.cudnn
import torch.optim
import torch.nn as nn
from tqdm import tqdm
from model import Model
from utils import *

warnings.filterwarnings('ignore', category=UserWarning)
model_names = sorted(name for name in Model.__dict__
    if name.islower() and not name.startswith("_")
    and callable(Model.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--store', default=False, help='store the log')
parser.add_argument('--exp_num', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--dataset', default='cifar100', help='dataset setting')
parser.add_argument('--pa', default=True, help='dataset setting')
parser.add_argument('--lam', default=0.5)
parser.add_argument('--t', default=1.)
# model config.
parser.add_argument('--arch', metavar='ARCH', default='resnet32', choices=model_names, help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--num_experts', default=3, help='the number of experts, equal to 1 denotes the plain model')
parser.add_argument('--loss_type', default="CE", choices=['CE', 'LDAM', 'Focal', 'BSL'], type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')

parser.add_argument('--head_shot_thr', default=100, help='decide the head class threshold')
parser.add_argument('--med_shot_thr', default=20, help='decide the med class threshold')

# hyperparameter config.
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',help='mini-batch size')
parser.add_argument('--epochs', default=200, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--lr_decay_epochs', default=[160, 180], help='when to decay lr')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--warmup', default=True, help='use warmup')
#parser.add_argument('--warmup_factor', type=float, default=0.1)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--cosine_anneal', default=False, help='cosine_annealing learning strategy')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')

parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=True, type=int, help='use GPU.')
parser.add_argument('--root_path', default='../datasets', type=str)

def main():
    args = parser.parse_args()
       
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    
    best_acc1 = 0

    _dataloader = imbalance_cifar.Cifar(args)
    train_loader, val_loader, cls_num_list, class_mapping = _dataloader.train_loader, _dataloader.test_loader, _dataloader.cls_num_list, _dataloader.class_mapping

    args.num_classes = len(cls_num_list)
    args.cls_num_list = np.array(sorted(cls_num_list, reverse=True))
    args.class_mapping = np.array(class_mapping)
    
    # set head and med class threshold for long-tailed datasets
    # threshold < 1 denote the head/med class percent
    if args.head_shot_thr < 1. and args.med_shot_thr < 1.:
        args.head_ave = int(len(cls_num_list) * args.head_shot_thr)
        args.med_ave = int(len(cls_num_list) * (1-args.med_shot_thr))
        args.head_shot_thr = args.cls_num_list[args.head_ave]
        args.med_shot_thr = args.cls_num_list[args.med_ave]
 
    # threshold > 1 denote the head/tail class number
    else:        
        # indices to split head, medium and tail classes
        args.head_ave = sum(args.cls_num_list >= args.head_shot_thr)
        args.med_ave = sum(args.cls_num_list >= args.med_shot_thr)

    # create model
    print("=> creating model '{}'".format(args.arch))
    
    criterion = CrossEntropyLoss().cuda()
    model = Model(backbone=args.arch, 
                  num_classes=args.num_classes, 
                  num_experts=args.num_experts)
    
    if args.gpu:
        model = model.cuda()
        
    if ngpus_per_node > 1:
        model = torch.nn.DataParallel(model).cuda()
    
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=args.weight_decay)
  
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # store result
    args.store_name = '_'.join([args.dataset, args.arch, str(args.num_experts), str(args.imb_factor), args.exp_num])
    if args.store:
        prepare_folders(args)
        with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
            f.write(str(args))
        f.close()

    print("=> start training  ")

    for epoch in range(args.start_epoch, args.epochs+1):
        lr = adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_acc = validate(val_loader, model, criterion, epoch, args)

        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)
        print('Best Prec@1: {:.3f} LR:{}'.format(best_acc1, lr))
        
        if args.store:
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                }, is_best)

            with open(os.path.join(args.root_log, args.store_name, 'log.txt'), 'a') as f:
                f.write('Best Prec@1: {:.3f} LR:{}\n'.format(best_acc1, lr))
                f.write('=====================\n')
            f.close()


    return model
    
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    am_con_loss = AverageMeter('Loss', ':.4e')
    am_kd_loss = AverageMeter('Loss', ':.4e')
    
    head = AverageMeter('Head', ':6.2f')
    medium = AverageMeter('Medium', ':6.2f')
    tail = AverageMeter('Tail', ':6.2f')

    e1 = AverageMeter('Acc@1', ':6.2f')
    e1_head = AverageMeter('Head', ':6.2f')
    e1_medium = AverageMeter('Medium', ':6.2f')
    e1_tail = AverageMeter('Tail', ':6.2f')
    
    e2 = AverageMeter('Acc@1', ':6.2f')
    e2_head = AverageMeter('Head', ':6.2f')
    e2_medium = AverageMeter('Medium', ':6.2f')
    e2_tail = AverageMeter('Tail', ':6.2f')

    e3 = AverageMeter('Acc@1', ':6.2f')
    e3_head = AverageMeter('Head', ':6.2f')
    e3_medium = AverageMeter('Medium', ':6.2f')
    e3_tail = AverageMeter('Tail', ':6.2f')    
    
    train_time = 0
    # switch to train mode
    model.train()
    pipeline = tqdm(train_loader)
    end = time.time()

    NTKD = NTKDLoss(args.t).cuda()
    NCE = NCELoss(args.num_classes, args.cls_num_list)

    for i, (inputs, indice) in enumerate(pipeline, start=1):
        # Sort categories by frequency and assign new category labels
        targets = torch.tensor([args.class_mapping[i] for i in indice], dtype=torch.int64)
        
        # measure data loading time
        data_time.update(time.time() - end)
        batchSize = targets.size(0)

        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)        

        # compute output
        logits, sim_logit = model(inputs, target = targets, args=args, mode='train')
        outputs = torch.stack(logits, dim=1).mean(1)
        logit_h, logit_m, logit_t = logits

        weight_h = F.one_hot(targets, args.num_classes).sum(dim=1)
        weight_m = F.one_hot(targets, args.num_classes)[:, args.head_ave:].sum(dim=1) 
        weight_t = F.one_hot(targets, args.num_classes)[:, args.med_ave:].sum(dim=1)      
        
        loss_h = criterion(logit_h, targets, weight_h) 
        loss_m = criterion(logit_m, targets, weight_m) 
        loss_t = criterion(logit_t, targets, weight_t) 
        cls_loss = loss_h +  loss_m + loss_t

        kd_loss = NTKD(logit_m, logit_h, 0, args.head_ave) + NTKD(logit_t, logit_h, 0, args.head_ave) +\
                  NTKD(logit_t, logit_m, args.head_ave, args.med_ave) 

        con_loss = NCE(sim_logit, targets)
 
        if epoch <= args.warmup_epochs:
            loss = cls_loss 
        else:
            loss = cls_loss + args.lam*(kd_loss + con_loss)
        
        # measure accuracy and record loss
        acc1, _ = Accuracy(outputs, targets, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        train_time += batch_time.val + data_time.val
        pipeline.set_description(
                'Epoch:{:3d}/{} Loss:{:.4f}'.format(epoch, args.epochs, loss.item()/batchSize))


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    head = AverageMeter('Head', ':6.2f')
    medium = AverageMeter('Medium', ':6.2f')
    tail = AverageMeter('Tail', ':6.2f')    

    e1 = AverageMeter('Acc@1', ':6.2f')
    e1_head = AverageMeter('Head', ':6.2f')
    e1_medium = AverageMeter('Medium', ':6.2f')
    e1_tail = AverageMeter('Tail', ':6.2f')
    
    e2 = AverageMeter('Acc@1', ':6.2f')
    e2_head = AverageMeter('Head', ':6.2f')
    e2_medium = AverageMeter('Medium', ':6.2f')
    e2_tail = AverageMeter('Tail', ':6.2f')

    e3 = AverageMeter('Acc@1', ':6.2f')
    e3_head = AverageMeter('Head', ':6.2f')
    e3_medium = AverageMeter('Medium', ':6.2f')
    e3_tail = AverageMeter('Tail', ':6.2f')    


    val_time = 0
    # switch to evaluate mode
    model.eval()

    pipeline = tqdm(val_loader)
    end = time.time()
    
    with torch.no_grad():
        for i, (inputs, indice) in enumerate(pipeline, start=1):
            # Sort categories by frequency and assign new category labels
            targets = torch.tensor([args.class_mapping[i] for i in indice], dtype=torch.int64)
            
            data_time.update(time.time() - end)
            batchSize = targets.size(0)

            inputs = inputs.cuda(non_blocking=True)    
            targets = targets.cuda(non_blocking=True)   
            # compute output
            logits, w  = model(inputs, target=targets, args=args)
            if args.pa:
                logits = [w[:, i] * logits[i].cuda() for i in range(args.num_experts)]     
       
            logit_h, logit_m, logit_t = logits
            outputs = torch.stack([logit_h, logit_m, logit_t], dim=1).sum(1)
            outputs = F.softmax(outputs, dim=1)

            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = Accuracy(outputs, targets, topk=(1, 5))
            head_acc, head_sample_num, mid_acc, mid_sample_num, tail_acc, tail_sample_num = Accuracy_Diffshot(outputs, targets, args)
            
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
    
            head.update(head_acc[0], head_sample_num)      
            medium.update(mid_acc[0], mid_sample_num)    
            tail.update(tail_acc[0], tail_sample_num)    
 
            #--------------------------------------
            logit_h = F.softmax(logit_h.detach(), dim=1)
            logit_m = F.softmax(logit_m.detach(), dim=1)
            logit_t = F.softmax(logit_t.detach(), dim=1)
            
            e1_acc, _ = Accuracy(logit_h, targets, topk=(1, 5))     # expert 1
            e2_acc, _ = Accuracy(logit_m, targets, topk=(1, 5))     # expert 2
            e3_acc, _ = Accuracy(logit_t, targets, topk=(1, 5))     # expert 3
        
            e1.update(e1_acc[0], targets.size(0))
            e2.update(e2_acc[0], targets.size(0))
            e3.update(e3_acc[0], targets.size(0))
            
            # expert 1
            head_acc, head_sample_num, mid_acc, mid_sample_num, tail_acc, tail_sample_num = Accuracy_Diffshot(logit_h, targets, args)
            e1_head.update(head_acc[0], head_sample_num)      
            e1_medium.update(mid_acc[0], mid_sample_num)    
            e1_tail.update(tail_acc[0], tail_sample_num)  
            
            # expert 2
            head_acc, head_sample_num, mid_acc, mid_sample_num, tail_acc, tail_sample_num = Accuracy_Diffshot(logit_m, targets, args)
            e2_head.update(head_acc[0], head_sample_num)      
            e2_medium.update(mid_acc[0], mid_sample_num)    
            e2_tail.update(tail_acc[0], tail_sample_num) 
            
            # expert 3
            head_acc, head_sample_num, mid_acc, mid_sample_num, tail_acc, tail_sample_num = Accuracy_Diffshot(logit_t, targets, args)
            e3_head.update(head_acc[0], head_sample_num)      
            e3_medium.update(mid_acc[0], mid_sample_num)    
            e3_tail.update(tail_acc[0], tail_sample_num)     
            #-------------------------------------

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(outputs, 1)
            
            pipeline.set_description(
                'Epoch:{:3d}/{} Loss:{:.4f}'.format(epoch, args.epochs, loss.item()/batchSize))
            val_time += batch_time.val + data_time.val
            
        print('Epoch:{} Val_Loss:{:.3f} Val_Top1:{:.3f} Head:{:.3f}  Medium:{:.3f}  Tail:{:.3f}  Val_Time:{:.3f}'.format(epoch, losses.avg, top1.avg, 
                                                                        head.avg, medium.avg, tail.avg, val_time))
        print('E1:', e1.avg.cpu(), e1_head.avg.cpu(), e1_medium.avg.cpu(), e1_tail.avg.cpu())
        print('E2:', e2.avg.cpu(), e2_head.avg.cpu(), e2_medium.avg.cpu(), e2_tail.avg.cpu())
        print('E3:', e3.avg.cpu(), e3_head.avg.cpu(), e3_medium.avg.cpu(), e3_tail.avg.cpu())
        
	  
        if args.store:
            with open(os.path.join(args.root_log, args.store_name, 'log.txt'), 'a') as f:
                f.write('Epoch:{} Val_Loss:{:.3f} Val_Top1:{:.3f} Head:{:.3f}  Medium:{:.3f}  Tail:{:.3f}  Val_Time:{:.3f}\n'.format(epoch, losses.avg, top1.avg, head.avg, medium.avg, tail.avg, val_time))
                f.write('E1: {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(e1.avg.cpu(), e1_head.avg.cpu(), e1_medium.avg.cpu(), e1_tail.avg.cpu()))
                f.write('E2: {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(e2.avg.cpu(), e2_head.avg.cpu(), e2_medium.avg.cpu(), e2_tail.avg.cpu()))
                f.write('E3: {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(e3.avg.cpu(), e3_head.avg.cpu(), e3_medium.avg.cpu(), e3_tail.avg.cpu()))   
            f.close()
            
    return top1.avg

class NCELoss(nn.Module):

    def __init__(self, num_classes, cls_num_list, beta=0.99):
        super(NCELoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.cls_num_list = cls_num_list

        
    def forward(self, logit, label):
        
        spc = torch.FloatTensor(self.cls_num_list).cuda()
        spc = spc.unsqueeze(0).expand(logit.shape[0], -1)
        logit = logit + spc.log()        
              
        proxy_label = torch.arange(0, self.num_classes).cuda()
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, proxy_label.T).float().cuda() #bz, cls

        # for numerical stability
        logits_max, _ = torch.max(logit, dim=1, keepdim=True)
        logits = logit - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) 

        # loss
        #weight = self.per_cls_weights[label].squeeze()
        loss = - mean_log_prob_pos.mean()
        return loss

class NTKDLoss(nn.Module):
    def __init__(self, t):
        super(NTKDLoss, self).__init__()
        self.t = t

    def forward(self, logit_s, logit_t, st_ave, ed_ave):

        batchSize, n = logit_t.size()
        mask = torch.zeros_like(logit_t).bool()
        mask[:, st_ave: ed_ave] = 1

        pred_s = F.log_softmax(logit_s[mask].view(batchSize, -1) / self.t, dim=1)
        pred_t = F.softmax(logit_t[mask].view(batchSize, -1) / self.t, dim=1)

        loss = F.kl_div(pred_s, pred_t, reduction='batchmean') * (self.t**2)
    
        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False, **kwargs):
        super().__init__()

        self.cls_num_list = cls_num_list
        if reweight_CE:
            idx = 1  # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = 1.
    
    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)

        return self

    def forward(self, logit, target, mask=1.):  
        num_classes = logit.size(1)
        target = nn.functional.one_hot(target, num_classes)
        log_logit = torch.log(F.softmax(logit, dim=1) + 1e-6)
        loss = - (mask * (target * log_logit).sum(dim=1)).mean()
        return loss        

if __name__ == '__main__':

    model = main()