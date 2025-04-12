import torch
import torch.nn as nn

def get_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

def build_optimizer(args, model, init_method='orthogonal'):
    model_unwrapped = get_model(model)
    ve_params = list(map(id, model_unwrapped.visual_extractor.parameters()))
    text_params = [] 
    params_to_optimize = []
    
    for name, param in model.named_parameters():
        if id(param) not in ve_params:
            if 'agent1' in name or 'agent2' in name:
                param.requires_grad = True
            elif 'tencoder.encoder' in name:  
                text_params.append(param)
            else:
                params_to_optimize.append(param)
                
            if init_method == 'orthogonal':
                if param.dim() > 1:
                    nn.init.orthogonal_(param, gain=1)
       
            
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model_unwrapped.visual_extractor.parameters(), 'lr': args.lr_ve}, 
         {'params': params_to_optimize, 'lr': args.lr_ed},
         {'params': text_params, 'lr': args.lt_ed}
        ], 
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler