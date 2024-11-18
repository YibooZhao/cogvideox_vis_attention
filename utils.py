import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers.models import CogVideoXTransformer3DModel
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from custom_processor import *

def cross_attn_init():
    CogVideoXAttnProcessor2_0.__call__ = attn__call__


def hook_fn(name,detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            timestep = module.processor.timestep
            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_maps[timestep][name] = module.processor.attn_map.cpu()
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        names = name.split('.')
        if not names[-1].startswith('attn1'):
            continue

        elif isinstance(module.processor, CogVideoXAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_fn(name))
    
    return unet


def set_layer_with_name_and_path(model, target_name="attn1", current_path=""):
    if model.__class__.__name__ == 'CogVideoXTransformer3DModel':
        model.forward = CogVideoXTransformer3DModelForward.__get__(model, CogVideoXTransformer3DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'CogVideoXBlock':
            layer.forward = CogVideoXBlockForward.__get__(layer, CogVideoXBlock)
        
        new_path = current_path + '.' + name if current_path else name
        set_layer_with_name_and_path(layer, target_name, new_path)
    
    return model


def split_attention_map_to_frames(attn_map, num_frames=3, attn_height=30,attn_weight=45):
    attention_maps = []
    frame_size = int(attn_height * attn_weight)

    for i in range(num_frames):
        for j in range(num_frames):
            start_h, end_h = i * frame_size, (i + 1) * frame_size
            start_w, end_w = j * frame_size, (j + 1) * frame_size
            frame = attn_map[start_h:end_h, start_w:end_w]

            reshaped_frame = frame.view(attn_height, attn_weight, frame_size)
            reduced_frame = reshaped_frame.mean(dim=2)
            
            attention_maps.append(((i, j), reduced_frame))

    return attention_maps


def enhance_contrast(tensor, factor=2.0):
    return tensor ** factor 

def save_attention_map(attn_map, file_path,name):
    attn_map = attn_map.to(torch.float32)
    plt.imshow(attn_map.cpu().numpy(), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(name)
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def normalize_min_max(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

def save_attention_map_as_tensor(attn_map,timestep,pt_path='pt'):
    os.makedirs(pt_path, exist_ok=True)
    file_path = os.path.join(pt_path, f"{timestep}.pt")
    torch.save(attn_map, file_path)

def resize_and_save(timestep=None, path=None, save_path='attn_maps',save_tensors='pt',latent_frame=None,attn_height=None,attn_width=None):
    resized_map = None

    if path is None:
        for path_ in list(attn_maps[timestep].keys()):
            value = attn_maps[timestep][path_]
            resized_map = resized_map + value if resized_map is not None else value   

        resized_map = normalize_min_max(resized_map)
        resized_map = enhance_contrast(resized_map)

        attention_maps = split_attention_map_to_frames(resized_map, latent_frame,attn_height,attn_width)

        os.makedirs(save_path, exist_ok=True)
        for (i, j),frame in attention_maps:
            name = f"{timestep}_{i}_{j}"
            file_path = os.path.join(save_path, f"{name}.png")
            save_attention_map(frame, file_path,name)
            
        if save_tensors != None:
            save_attention_map_as_tensor(resized_map,timestep,save_tensors)

def save_by_timesteps(latent_frame,attn_height,attn_width,save_path='attn_maps_by_timesteps',save_tensors=None):
    for timestep in tqdm(attn_maps.keys(),total=len(list(attn_maps.keys()))):
        resize_and_save(timestep, None, save_path,save_tensors,latent_frame,attn_height,attn_width)

