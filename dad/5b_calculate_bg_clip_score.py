import glob
import json
import numpy as np
import os
import shutil


import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from pycocotools.coco import COCO  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
import clip

from vis_prior.utils import *


RAW_PROMPT = lambda prompt: prompt
A_PROMPT = lambda prompt: "a " + prompt
P_PROMPT = lambda prompt: "picture of a " + prompt
PROMPT_ENGINEERING = {
    "l": {
        "csl": RAW_PROMPT,
        "csl_a": A_PROMPT,
        "csl_p": P_PROMPT,
    },
    "b": {
        "csb": RAW_PROMPT,
        "csb_a": A_PROMPT,
        "csb_p": P_PROMPT,
    },
}


def calculate_bg_clip_score(
    unfiltered_data_folder,
    clip_mode,
    device,
    mask_fill_color="black",
    prompt_mode="csl",
):

    assert clip_mode in ["l", "b"]
    if clip_mode == "l":
        clip_model_name =  "ViT-L/14"
    elif clip_mode == "b":
        clip_model_name =  "ViT-B/32"
    else:
        raise ValueError(f"clip_mode should be in {['l', 'b']}, but it is: {clip_mode}")
    
    assert prompt_mode in PROMPT_ENGINEERING[clip_mode].keys()
    pe = PROMPT_ENGINEERING[clip_mode][prompt_mode]

    unfiltered_annotation_path = os.path.join(unfiltered_data_folder, "annotation.json")

    with open(unfiltered_annotation_path, "r") as f:
        unfiltered_annotation = json.load(f)
    coco = COCO(unfiltered_annotation_path)

    model, preprocess = clip.load(clip_model_name, device=device)

    for img_obj in tqdm(unfiltered_annotation['images']):
        image_id = img_obj["id"]
        img_path = os.path.join(unfiltered_data_folder, 'images', img_obj['file_name'])
        img = Image.open(img_path)
        
        ann_ids = coco.getAnnIds(imgIds=image_id)
        imgdraw = ImageDraw.Draw(img)  
        cat_ids = set()
        for ann_id in ann_ids:
            ann = coco.loadAnns([ann_id])[0]
            x, y, w, h = ann['bbox']
            imgdraw.rectangle((x, y, x+w, y+h), fill=mask_fill_color)
            cat_ids.add(ann['category_id'])
        
        cat_objs = coco.loadCats(cat_ids)
        cat_names = [cat_obj["name"] for cat_obj in cat_objs] # only detect for categories assigned
        
        image = preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize([pe(prompt) for prompt in cat_names]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            
        for i, cat_name in enumerate(cat_names):
            img_obj["%s_%s"%(prompt_mode,cat_name)] = float(logits_per_image[0][i])
    
    with open(unfiltered_annotation_path, "w+") as f:
        json.dump(unfiltered_annotation, f)

    print(len(unfiltered_annotation['images']))
    print(unfiltered_annotation['images'][222])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--unfiltered_data_folder", default=None, type=str)
    parser.add_argument("--clip_mode", default="l", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--mask_fill_color", default="black", type=str)
    parser.add_argument("--prompt_mode", default="csl", type=str)
    args = parser.parse_args()

    '''
    e.g. coco10novel (coco 10 shot, novel cat only)
    python3 5_calculate_ann_clip_score.py \
        -d /media/data/dad/cnet/experiments/coco10novel/mix_n2000_o1_s1_p640_promptenhanced 
    '''

    calculate_bg_clip_score(
            unfiltered_data_folder=args.unfiltered_data_folder,
            clip_mode=args.clip_mode,
            device=args.device,
            mask_fill_color=args.mask_fill_color,
            prompt_mode=args.prompt_mode,
        )


if __name__ == "__main__":
    main()
