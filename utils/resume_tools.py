import torch
import timm
import torchvision

def resume_lightning(model, weight_path):
    state_dict = torch.load(weight_path, map_location='cpu')['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # remove `model.` from key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)




