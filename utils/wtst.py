import torch

from utils._utils import *
from utils import transformers
from utils.transformers import *
from torchvision import models as cnn_models
from torchvision.models import ResNet, Inception3


def shuffle_weights(weight):
    # applies random shuffling of the weights, i.e. sampling without replacement
    idx = torch.randperm(weight.nelement())
    new_wight = weight.view(-1)[idx].view(weight.size())
    return new_wight


def sample_from_weight_dist(weight):
    n_elems = weight.nelement()
    idx = np.random.choice(n_elems, size=n_elems, replace=True)
    new_weight = weight.view(-1)[idx].view(weight.size())
    return new_weight


def sampled_iid_random(weight, mean=0, std=0.05):
    return torch.normal(mean=mean, std=std, size=weight.shape)


def sample_iid_gaussian(weight):
    mean, std = torch.mean(weight), torch.std(weight)
    new_weight = torch.normal(mean=mean.item(), std=std.item(), size=weight.shape)
    return new_weight

def init_bn_module(submodel):
    """
    Given a BN module it returns a 01 initialized state
    """
    for layer in submodel.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.reset_parameters()  

def init_ln_module(submodel):
    """
    Given a Layer Norm module it returns a 01 initialized state
    """
    for layer in submodel.modules():
        if isinstance(layer, nn.LayerNorm):
            layer.reset_parameters()

            
def get_wt_state(model, sampling_fn=sample_iid_gaussian):
    # detach original state
    orig_state = deepcopy(model.state_dict())

    # init Layer Norm modules
    init_ln_module(model)    
    # get current state
    state = model.state_dict()
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            weight_name = f"{name}.weight"
            bias_name = f"{name}.bias"
            state[weight_name] = sampling_fn(state[weight_name])
            if bias_name in state:
                state[bias_name] = sampling_fn(state[bias_name])

    # reverting model to the original state and return WT state
    state = deepcopy(state)
    model.load_state_dict(orig_state) 
    return state


def cnn_init(model, transfer_up_to_layer=-2, sampling_fn=sample_iid_gaussian,
                use_checkpoint="", return_original=False):
    """
    Initialized an ImageNet-pretrained model and resamples/shuffles weights if wanted.
    :param transfer_up_to_layer: -2 to N where N=number of layers to use WT.
    :param sampling_fn: determines the sampling function for the new weights.
    :return: model, orig_model where the first one is the WT init of the second one. The second one (actual ImageNet
    initialized) is returned just in case we need it.
    """

    inception_names = {"layer-1" : ["Conv2d_1a_3x3"],
                       "layer0" : ["Conv2d_2a_3x3", "Conv2d_2b_3x3"],
                       "layer1" : ["Conv2d_3b_1x1"],
                       "layer2" : ["Conv2d_4a_3x3"],
                       "layer3" : ["Mixed_5b", "Mixed_5c", "Mixed_5d"],
                       "layer4" : ["Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e"],
                       "layer5" : ["Mixed_7a", "Mixed_7b", "Mixed_7c"]}
    
    
    model_type = model.__class__.__name__
    if not isinstance(model, (ResNet, Inception3)):        
        raise NotImplementedError("This function does not support {} models".format(model_type))
    print_ddp("\033[94m WT initialization\033[0m with transfer_up_to_layer={}".format(transfer_up_to_layer))
    
    # getting model's state
    state = model.state_dict()
    if return_original:
        orig_state = deepcopy(state)

    # if a checkpoint path is present (use_checkpoint != "")
    # checks if checkpoint path exists. If it does not it saves the current state
    # Otherwise, it loads the state from the checkpoint
    if use_checkpoint != "" and os.path.isfile(use_checkpoint):
        wt_state = torch.load(use_checkpoint)
        print_ddp("Loading WT checkpoint from {}".format(use_checkpoint))
    else:
        wt_state = get_wt_state(model, sampling_fn=sampling_fn)
        print_ddp("Creating new WT checkpoint at {}".format(use_checkpoint))
        torch.save(wt_state, use_checkpoint)        
        
    # the first conv layer
    if transfer_up_to_layer < -1:  # if -1 or higher, transfer the weights of conv1, otherwise resample
        if isinstance(model, ResNet): 
            state['conv1.weight'] = wt_state['conv1.weight']
            if 'conv1.bias' in state:
                state['conv1.bias'] = wt_state['conv1.bias']
            print_ddp('---- Performed WT init for conv1')
        elif isinstance(model, Inception3): 
            for blk in inception_names["layer-1"]:
                init_bn_module(getattr(model, blk))
                for name, module in getattr(model, blk).named_modules():
                    if isinstance(module, nn.Conv2d):
                        weight_name = f"{blk}.{name}.weight"
                        bias_name = f"{blk}.{name}.bias"
                        state[weight_name] = wt_state[weight_name]
                        if bias_name in state:
                            state[bias_name] = wt_state[bias_name] 
            print_ddp(f'---- Performed WT init for layer-1 : {inception_names["layer-1"]}')
        else:
            raise NotImplementedError(f"This function does not support {model_type} models")

    if transfer_up_to_layer < 0:  # layer 0 or higher: keep BatchNorm as well
        if isinstance(model, ResNet): 
            init_bn_module(model.bn1)
            print_ddp('---- Performed WT init for bn1')
        elif isinstance(model, Inception3): 
            for blk in inception_names["layer0"]:
                init_bn_module(getattr(model, blk))
                for name, module in getattr(model, blk).named_modules():
                    if isinstance(module, nn.Conv2d):
                        weight_name = f"{blk}.{name}.weight"
                        bias_name = f"{blk}.{name}.bias"
                        state[weight_name] = wt_state[weight_name]
                        if bias_name in state:
                            state[bias_name] = wt_state[bias_name]    
            print_ddp(f'---- Performed WT init for layer0 : {inception_names["layer0"]}')
        else:
            raise NotImplementedError(f"This function does not support {model_type} models")                        

    
    if isinstance(model, ResNet): 
        layer_n = 5 
    elif isinstance(model, Inception3):  
        layer_n = 6
    else:
        raise NotImplementedError(f"This function does not support {model_type} models") 
        
    for layer in list(range(1, layer_n)):
        if layer <= transfer_up_to_layer:  # keeping all layer lower or equal to transfer_up_to_layer
            continue
        curr_layer = f'layer{layer}' 
        
        if isinstance(model, ResNet): 
            init_bn_module(getattr(model, curr_layer))
            for name, module in getattr(model, curr_layer).named_modules():
                if isinstance(module, nn.Conv2d):
                    weight_name = "{}.{}.weight".format(curr_layer, name)
                    bias_name = "{}.{}.bias".format(curr_layer, name)
                    state[weight_name] = wt_state[weight_name]
                    if bias_name in state:
                        state[bias_name] = wt_state[bias_name]
            print_ddp(f'---- Performed WT init for {curr_layer}')
            
        elif isinstance(model, Inception3): 
            for blk in inception_names[curr_layer]:
                init_bn_module(getattr(model, blk))
                for name, module in getattr(model, blk).named_modules():
                    if isinstance(module, nn.Conv2d):
                        weight_name = f"{blk}.{name}.weight"
                        bias_name = f"{blk}.{name}.bias"
                        state[weight_name] = wt_state[weight_name]
                        if bias_name in state:
                            state[bias_name] = wt_state[bias_name]     
            print_ddp(f'---- Performed WT init for {curr_layer} : {inception_names[curr_layer]}')
        else:
            raise NotImplementedError(f"This function does not support {model_type} models")             

    # loading updated state_dict
    model.load_state_dict(state)

    if return_original:
        return orig_state
    

def transformer_init(model, transfer_up_to_layer=-2, sampling_fn=sample_iid_gaussian,
                     use_checkpoint="", return_original=False):
    """
    Initialized an ImageNet-pretrained model and resamples/shuffles weights if wanted.
    :param transfer_up_to_layer: -2 to N where N=number of layers to use WT.
    :param sampling_fn: determines the sampling function for the new weights.
    :return: model, orig_model where the first one is the WT init of the second one. The second one (actual ImageNet
    initialized) is returned just in case we need it.
    """
    model_type = model.__class__.__name__
    if not isinstance(model, (VisionTransformer, SwinTransformer, FocalTransformer)):
        raise NotImplementedError("This function does not support {} models".format(model_type))
    print_ddp("\033[94m WT initialization\033[0m with transfer_up_to_layer={}".format(transfer_up_to_layer))
   
    # getting model's state
    state = model.state_dict()
    if return_original:
        orig_state = deepcopy(state)

    # if a checkpoint path is present (use_checkpoint != "")
    # checks if checkpoint path exists. If it does not it perfroms WT and saves the current state
    # Otherwise, it loads the state from the checkpoint
    if use_checkpoint != "" and os.path.isfile(use_checkpoint):
        wt_state = torch.load(use_checkpoint)
        print_ddp("Loading WT checkpoint from {}".format(use_checkpoint))
    else:
        wt_state = get_wt_state(model, sampling_fn=sampling_fn)
        print_ddp("Creating new WT checkpoint at {}".format(use_checkpoint))
        torch.save(wt_state, use_checkpoint)     
        
    # the first layer
    if transfer_up_to_layer < -1:  # if -2, resample tokenizer, otherwise transfer it
        init_ln_module(model.patch_embed)
        if 'cls_token' in state:
            state['cls_token'] = wt_state['cls_token']
        if 'pos_embed' in state:
            state['pos_embed'] = wt_state['pos_embed']
        state['patch_embed.proj.weight'] = wt_state['patch_embed.proj.weight']
        if 'patch_embed.proj.bias' in state:
            state['patch_embed.proj.bias'] = wt_state['patch_embed.proj.bias']
        print_ddp('---- Performed WT init for patchifier')

    layer_n = -1
    if isinstance(model, VisionTransformer):
        for layer in list(range(len(model.blocks))):
            # keeping all layer lower or equal to transfer_up_to_layer
            if layer <= transfer_up_to_layer:
                continue
            curr_layer = model.blocks[layer]
            init_ln_module(curr_layer)

            for name, module in curr_layer.named_modules():
                if isinstance(module, nn.Linear):
                    weight_name = "blocks.{}.{}.weight".format(layer, name)
                    bias_name = "blocks.{}.{}.bias".format(layer, name)
                    state[weight_name] = wt_state[weight_name]
                    state[bias_name] = wt_state[bias_name]
            print_ddp(f'---- Performed WT init for block{layer}')
            
    elif isinstance(model, (SwinTransformer, FocalTransformer)):
        for layer in list(range(len(model.layers))):
            curr_layer = model.layers[layer]
            for blk in list(range(len(curr_layer.blocks))):
                layer_n += 1
                if layer_n <= transfer_up_to_layer:
                    continue  
                    
                init_ln_module(curr_layer.blocks[blk])
                for name, module in curr_layer.blocks[blk].named_modules():
                    if isinstance(module, nn.Linear):
                        weight_name = f"layers.{layer}.blocks.{blk}.{name}.weight"
                        bias_name = f"layers.{layer}.blocks.{blk}.{name}.bias"
                        state[weight_name] = wt_state[weight_name]
                        state[bias_name] = wt_state[bias_name] 
                print_ddp(f'---- Performed WT init for layer-{layer} block-{blk}')
                
            if hasattr(curr_layer, "downsample") and transfer_up_to_layer <= layer_n:
                if curr_layer.downsample is None: continue
                init_ln_module(curr_layer.downsample)
                for name, module in curr_layer.downsample.named_modules():
                    if isinstance(module, nn.Linear):
                        weight_name = f"layers.{layer}.downsample.{name}.weight"
                        bias_name = f"layers.{layer}.downsample.{name}.bias"
                        state[weight_name] = wt_state[weight_name]
                        if bias_name in state:
                            state[bias_name] = wt_state[bias_name]
    else:
        raise NotImplementedError(f"This function does not support {model_type} models")
                                                      
        
    if isinstance(model, VisionTransformer):
        _norm_init = transfer_up_to_layer <= len(model.blocks) - 1
    elif isinstance(model, (SwinTransformer, FocalTransformer)):
        _norm_init = transfer_up_to_layer <= layer_n

    if model.norm is not None and _norm_init:
        init_ln_module(model.norm)
        print_ddp('---- Performed WT init for the last norm layer')

    # loading updated state_dict
    model.load_state_dict(state)
    if return_original:
        return orig_state    
    

def wtst_init(model, transfer_up_to_layer,
                     sampling_fn=sample_iid_gaussian,
                     use_checkpoint=""):    
    
    model_type = model.__class__.__name__
    if hasattr(cnn_models, model_type):
        cnn_init(model, 
                    transfer_up_to_layer=transfer_up_to_layer, 
                    sampling_fn=sampling_fn,
                    use_checkpoint=use_checkpoint, 
                    return_original=False)
    elif hasattr(transformers, model_type):
        transformer_init(model,
                         transfer_up_to_layer=transfer_up_to_layer,
                         sampling_fn=sampling_fn,
                         use_checkpoint=use_checkpoint,
                         return_original=False)
    else:
        raise NotImplementedError("The current implementation does not support {} models".format(model_type))
