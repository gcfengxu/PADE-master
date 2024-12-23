import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from backbone import resnet_cifar
from backbone import expert_resnet_cifar

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    #@abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

class Model(BaseModel):

    def __init__(self, backbone, num_classes, num_experts,
                 reduce_dimension=False, use_norm=False, 
                 returns_feat=True, share_layer3=False):
        super().__init__()
        if backbone is not None: 
            self.backbone = eval(backbone)(num_classes=num_classes, 
                                           num_experts=num_experts, 
                                           use_norm=use_norm, 
                                           returns_feat=returns_feat,
                                           reduce_dimension=reduce_dimension)

        self.num_experts = num_experts
        self.num_classes = num_classes

        dim_mlp = self.backbone.linears[0].in_features if num_experts!=1 else self.backbone.linear.in_features  
        feat_dim = 256 
        self.encoder = nn.Sequential(nn.Linear(dim_mlp*num_experts, dim_mlp), 
                                     nn.BatchNorm1d(dim_mlp),
                                     nn.ReLU(True), 
                                     nn.Linear(dim_mlp, feat_dim)) 
        
        self.register_buffer('memory_pt', torch.randn(num_classes, feat_dim)) 
        self.memory_pt = F.normalize(self.memory_pt, dim=1)

        self.momentum = 0.9

    def _cal_similarity(self, feat, label):

        memory_pt = self.memory_pt.detach()

        for x, y in zip(feat, label):
            pt = self.momentum * memory_pt[y] + (1-self.momentum) * x
            memory_pt[y] = F.normalize(pt, dim=0)

        self.memory_pt = memory_pt
       
        # calculate logit of similarity  
        sim_logit = torch.einsum('nc, kc-> nk', [feat, self.memory_pt])

        return sim_logit 
    
    def _prototype_alignment(self, feat, target, args):
        
        sim = torch.einsum('nc, kc-> nk', [feat, self.memory_pt.clone().detach()])
        
        w_h, idx_h = torch.max(sim[:, : args.head_ave], dim=1)
        w_m, idx_m = torch.max(sim[:, args.head_ave: args.med_ave], dim=1)
        w_t, idx_t = torch.max(sim[:, args.med_ave: ], dim=1)

        w_h, w_m, w_t = w_h.unsqueeze(1), w_m.unsqueeze(1), w_t.unsqueeze(1)
        
        w = torch.cat([w_h, w_m, w_t], dim=1)
        w = w / torch.sum(w, dim=1, keepdim=True)

        return w.unsqueeze(-1)
    
    def _hook_before_iter(self):
        
        self.backbone._hook_before_iter()
    
    def forward(self, x, target=None, args=None, mode='test'):

        x = self.backbone(x)

        feats, logits = x['feat'], x['logits']

        feat = torch.cat(feats, dim=1)
        feat = self.encoder(feat)
        feat = F.normalize(feat, dim=1)
        
        if mode == 'train':
            
            sim_logit = self._cal_similarity(feat, target)
            return logits, sim_logit
        
        if mode == 'test':
            w = self._prototype_alignment(feat, target, args).to('cuda:0')    

            return logits, w
        
    
def resnet32(num_classes, num_experts=1, use_norm=False, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, **kwargs):

    if num_experts == 1:
        return resnet_cifar.ResNet_s(resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, **kwargs)
    else:
        return expert_resnet_cifar.ResNet_s(expert_resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)


# From LDAM_DRW
def resnet32_b(num_classes, num_experts=1, use_norm=False, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, **kwargs):

    if num_experts == 1:
        return resnet_cifar.ResNet_s(resnet_cifar.BasicBlockB, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, **kwargs)
    else:
        return expert_resnet_cifar.ResNet_s(expert_resnet_cifar.BasicBlockB, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)
