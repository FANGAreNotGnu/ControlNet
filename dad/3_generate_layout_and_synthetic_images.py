from PIL import Image
from tqdm import tqdm
import argparse
import json
import matplotlib.pyplot as plt
import cv2
import einops
# import gradio as gr
import numpy as np
import torch
import os
import random

from vis_prior.utils import *
from vis_prior.vis_prior_layout import UniformRandomNoClipVPL
from vis_prior.vis_prior_generator import CannyVPG, HEDVPG, MLSDVPG, MidasVPG, UniformerVPG


import sys
sys.path.append("/home/ubuntu/dad/ControlNet")
from share import *
import config
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def process_hed(model, ddim_sampler, vis_prior, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        detected_map = HWC3(vis_prior.astype(np.uint8))
        H, W, C = detected_map.shape

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


def generate_layout_and_synthetic_images(
        annotation_path, 
        im_folder, 
        model_config_path, 
        experiment_path, 
        ckpt_name, 
        num_layouts, 
        num_object_per_layout, 
        num_samples_per_layout, 
        pixels_size,
        synthetic_data_postfix = None,
        vpl_mode = "naive",
        vpg_mode = "HED",
        vpl_scale_down = 1.,
        vpl_scale_up = 1.,
        vpl_min_ratio = -1,
    ):
    synthetic_data_save_name = f"syn_n{num_layouts}_o{num_object_per_layout}_s{num_samples_per_layout}_p{pixels_size}"
    if synthetic_data_postfix is not None:
        synthetic_data_save_name += "_" + synthetic_data_postfix
        
    ckpt_path = os.path.join(experiment_path, ckpt_name)
    synthetic_data_save_path = os.path.join(experiment_path, synthetic_data_save_name)
    if vpl_mode == "naive":
        vpl = UniformRandomNoClipVPL(scale_down = vpl_scale_down, scale_up = vpl_scale_up, min_ratio = vpl_min_ratio)
    else:
        raise ValueError(f"vpl_mode: {vpl_mode}")
    if vpg_mode == "HED":
        vpg = HEDVPG(vpl=vpl, fill_val=0, annotation=annotation_path, im_folder=im_folder)
    else:
        raise ValueError(f"vpg_mode: {vpg_mode}")

    layouts = vpg.generate_layouts(im_shape=(pixels_size,pixels_size,3), num_object_per_layout=num_object_per_layout, num_layouts=num_layouts)
    prompts = vpg.generate_prompts(layouts=layouts)
    vis_priors = vpg.generate_vis_priors(layouts=layouts, im_shape=(pixels_size,pixels_size,vpg.sample_channel))

    model = create_model(model_config_path).cpu()
    model.load_state_dict(load_state_dict(ckpt_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    if not os.path.exists(synthetic_data_save_path):
        os.mkdir(synthetic_data_save_path)
        
    for i in tqdm(range(num_layouts)):
        results_per_layout = process_hed(
                        model=model,
                        ddim_sampler=ddim_sampler,
                        vis_prior=vis_priors[i],
                        prompt="picture of a " + prompts[i],
                        a_prompt="realistic, real, photo", 
                        n_prompt="blurry, unclear, duplicate, distortion, lowres, bad anatomy, error, cropped, low quality", 
                        num_samples=num_samples_per_layout, 
                        ddim_steps=50, 
                        guess_mode=False, 
                        strength=1., 
                        scale=9., 
                        seed=1, 
                        eta=0.,
                    )[1:]
        #results.append(results_per_layout)

    # results: a (num_layouts, num_samples_per_layout) list of images
    # layouts: a (num_layouts,) list of layout
    # layout: [[category_name, bbox, (im_prior)], ...]
    #for i in range(num_layouts):
        synthetic_sample_save_path = os.path.join(synthetic_data_save_path, "%08d"%i)
        if not os.path.exists(synthetic_sample_save_path):
            os.mkdir(synthetic_sample_save_path)

        layout = layouts[i]
        layout_cats_save_path = os.path.join(synthetic_sample_save_path, "layout_cats.npy")
        layout_bboxes_save_path = os.path.join(synthetic_sample_save_path, "layout_bboxes.npy")
        layout_priors_save_path = os.path.join(synthetic_sample_save_path, "layout_priors.npy")

        cats = np.array([item[0] for item in layout])
        bboxes = np.array([item[1] for item in layout])
        priors = np.array([item[2] for item in layout])

        np.save(layout_cats_save_path, cats)
        np.save(layout_bboxes_save_path, bboxes)
        np.save(layout_priors_save_path, priors)

        # write vis_prior for this layout
        imwrite(os.path.join(synthetic_sample_save_path, "vis_prior.jpg"), vis_priors[i])

        # write synthetic images for this layout
        for j in range(num_samples_per_layout):
            imwrite(os.path.join(synthetic_sample_save_path, "syn%03d.jpg"%j), results_per_layout[j])

        # write prompt for this layout TODO: we'll have multiple prompt per sample
        prompt = prompts[i]
        prompt_save_path = os.path.join(synthetic_sample_save_path, "prompt.npy")
        np.save(prompt_save_path, [prompt])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotation_path", default=None, type=str)
    parser.add_argument("-i", "--im_folder", default=None, type=str)
    parser.add_argument("--model_config_path", default='/home/ubuntu/dad/ControlNet/models/cldm_v15.yaml', type=str)
    parser.add_argument("-e", "--experiment_path", default=None, type=str)
    parser.add_argument("-c", "--ckpt_name", default=None, type=str)
    parser.add_argument("-l", "--num_layouts", default=None, type=int)
    parser.add_argument("-o", "--num_object_per_layout", default=None, type=int)
    parser.add_argument("-s", "--num_samples_per_layout", default=None, type=int)
    parser.add_argument("-p", "--pixels_size", default=None, type=int)  # now we generate square images only
    parser.add_argument("--synthetic_data_postfix", default=None, type=str)
    parser.add_argument("--vpl_mode", default="naive", type=str)
    parser.add_argument("--vpg_mode", default="HED", type=str)
    parser.add_argument("-d", "--vpl_scale_down", default=1., type=float)
    parser.add_argument("-u", "--vpl_scale_up", default=1., type=float)
    parser.add_argument("-r", "--vpl_min_ratio", default=-1, type=float)

    '''
    e.g. coco-10shot-novelonly
    CUDA_VISIBLE_DEVICES=0 python3 3_generate_layout_and_synthetic_images.py \
        -a /media/data/coco17/coco/seed1/10shot_novel.json \
        -i /media/data/coco17/coco/train2017/ \
        -e /media/data/dad/cnet/experiments/coco10novel \
        -c "epoch=99-step=2999.ckpt" \
        -l 2000 \
        -o 1 \
        -s 1 \
        -p 640 \
        --synthetic_data_postfix promptenhanced

        
    e.g. coco-10shot-novelonly-scaled
    CUDA_VISIBLE_DEVICES=0 python3 3_generate_layout_and_synthetic_images.py \
        -a /media/data/coco17/coco/seed1/10shot_novel.json \
        -i /media/data/coco17/coco/train2017/ \
        -e /media/data/dad/cnet/experiments/coco10novel \
        -c "epoch=99-step=2999.ckpt" \
        -l 20 \
        -o 1 \
        -s 1 \
        -p 640 \
        -d 0.5 \
        -u 2 \
        -r 0.1 \
        --synthetic_data_postfix scaled
    '''

    args = parser.parse_args()

    generate_layout_and_synthetic_images(
        annotation_path=args.annotation_path, 
        im_folder=args.im_folder, 
        model_config_path=args.model_config_path, 
        experiment_path=args.experiment_path, 
        ckpt_name=args.ckpt_name, 
        num_layouts=args.num_layouts, 
        num_object_per_layout=args.num_object_per_layout, 
        num_samples_per_layout=args.num_samples_per_layout, 
        pixels_size=args.pixels_size, 
        synthetic_data_postfix=args.synthetic_data_postfix, 
        vpl_mode=args.vpl_mode, 
        vpl_scale_down=args.vpl_scale_down, 
        vpl_scale_up=args.vpl_scale_up, 
        vpl_min_ratio=args.vpl_min_ratio, 
    )   

if __name__ == "__main__":
    main()
