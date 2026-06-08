from functools import partial

import torch
from torch import nn

from lib.model.DSTformer import DSTformer
from lib.model.diffusionpose import *

def count_parameters(model):
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    return model_params


def load_pretrained_weights(model, checkpoint):
    """
    Load pretrained weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - checkpoint (dict): the checkpoint
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    model_first_key = next(iter(model_dict))
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if not 'module.' in model_first_key:
            if k.startswith('module.'):
                k = k.replace('module.','')
        if k in model_dict:
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print(f'[INFO] (load_pretrained_weights) {len(matched_layers)} layers are loaded')
    print(f'[INFO] (load_pretrained_weights) {len(discarded_layers)} layers are discared')
    if len(matched_layers) == 0:
        print ("--------------------------model_dict------------------")
        print (model_dict.keys())
        print ("--------------------------discarded_layers------------------")
        print (discarded_layers)
        raise NotImplementedError(f"Loading problem!!!!!!")



def load_pretrained_backbone(params, backbone_name):
    if backbone_name == 'motionbert':
        model_backbone = DSTformer(dim_in=3,
                                   dim_out=3,
                                   dim_feat=params['dim_feat'],
                                   dim_rep=params['dim_rep'],
                                   depth=params['depth'],
                                   num_heads=params['num_heads'],
                                   mlp_ratio=params['mlp_ratio'],
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   maxlen=params['maxlen'],
                                   num_joints=params['num_joints'])
        checkpoint = torch.load(params['model_checkpoint_path'], map_location=lambda storage, loc: storage)['model_pos']
    elif backbone_name == 'D3DP':
        backbone = DSTformer(dim_in=3,
                            dim_out=3,
                            dim_feat=params['dim_feat'],
                            dim_rep=params['dim_rep'],
                            depth=params['depth'],
                            num_heads=params['num_heads'],
                            mlp_ratio=params['mlp_ratio'],
                            norm_layer=partial(nn.LayerNorm, eps=1e-6),
                            maxlen=params['maxlen'],
                            num_joints=params['num_joints'])
        joints_left = [4,5,6,11,12,13]
        joints_right = [1,2,3,14,15,16]
        model_backbone = D3DP(params, joints_left, joints_right, backbone=backbone, is_train=False, sampling_timesteps=params['sampling_timesteps'])
        checkpoint = torch.load(params['model_checkpoint_path'], map_location=lambda storage, loc: storage)['model_pos']
    else:
        raise Exception("Undefined backbone type.")

    load_pretrained_weights(model_backbone, checkpoint)
    return model_backbone
