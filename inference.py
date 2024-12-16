from share import *
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import os
import torch
import cv2
import einops
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_metrics(image1, image2):
    
    psnr = peak_signal_noise_ratio(image1, image2, data_range=255)

    return psnr

def process(model, sampler, input_image, device):
    with torch.no_grad():
        
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
            
        control = input_image['hint']
        control = einops.rearrange(control, 'b h w c -> b c h w').clone().to(device)
        a_prompt = input_image["txt"][0]
        
        H= control.shape[2]
        W = control.shape[3]
        num_samples=2
        n_prompt = ""
        strength = 1.8
        ddim_steps = 20
        eta = 0
        scale = 7
        
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([a_prompt] * num_samples)]}
        un_cond = {"c_concat":[control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)
        

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)]  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    return x_samples


# resume_path = '/project/cigserver5/export1/david.w/MixViewDiff/ControlNet/lightning_logs/version_1/checkpoints/epoch=36-step=63084.ckpt'
resume_path = '/project/cigserver5/export1/a.peng/ControlNet/lightning_logs/from_david/epoch=157-step=202081.ckpt'
# resume_path = '/project/cigserver5/export1/david.w/MixViewDiff/ControlNet/lightning_logs/version_51/checkpoints/epoch=160-step=205918.ckpt'
batch_size = 2
device = "cuda:0"

print('resume_path:', resume_path)
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location=device))
model = model.cuda()
sampler = DDIMSampler(model)


dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)


# Iterate through dataloader and compare its ground truth to output of model
for idx, batch in enumerate(dataloader, start=0):
    gt = (batch["jpg"]* 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    samples = process(model, sampler, batch, device)
    
    for b in range(batch_size):
        image = Image.fromarray(samples[b])
        gt_im = Image.fromarray(gt[b])
        
        v = idx * batch_size + b
        
        image.save(f"./samples/{v}_sample.png")
        gt_im.save(f"./samples/{v}_gt.png")
        
        psnr= calculate_metrics(gt[b], samples[b])
        print(f"psnr: {psnr}")
        
    

