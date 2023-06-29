from collections import defaultdict
import argparse
import glob
import json
import numpy as np
import os
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb


def filter_background(
    unfiltered_data_folder,
    clip_score_key,
    percent_kept,
):
    unfiltered_annotation_path = os.path.join(unfiltered_data_folder, "annotation.json")
    unfiltered_image_folder = os.path.join(unfiltered_data_folder, "images")
    target_folder = unfiltered_data_folder + "_pfb_%s%d" % (clip_score_key, percent_kept)

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    target_image_folder = os.path.join(target_folder, "images")
    target_annotation_path = os.path.join(target_folder, "annotation.json")

    if not os.path.exists(target_image_folder):
        os.mkdir(target_image_folder)

    with open(unfiltered_annotation_path, "r") as f:
        unfiltered_annotation = json.load(f)
    coco = COCO(unfiltered_annotation_path)

    cat_names = [cat['name'] for cat in unfiltered_annotation["categories"]]

    clip_scores = defaultdict(list)
    for img_obj in unfiltered_annotation["images"]:
        if 'coco_url' not in img_obj.keys():  # only use synthetic data (no GT data) to set the threshold
            for k in img_obj.keys():
                if k[:len(clip_score_key)] == clip_score_key:  # <clip_score_key>_<cat_name>
                    cat_name = k[len(clip_score_key) + 1:]
                    clip_scores[cat_name].append(img_obj[k])

    csl_thres = {category_name:np.percentile(scores, percent_kept) for category_name, scores in clip_scores.items()}

    old_images = unfiltered_annotation["images"]
    old_annos = unfiltered_annotation["annotations"]
    new_annos = []
    new_images = []

    valid_image_ids = []
    for img_obj in old_images:
        is_valid_image = True
        if 'coco_url' not in img_obj.keys():
            for k in img_obj.keys():
                if k[:len(clip_score_key)] == clip_score_key:  # csl_<cat_name>
                    cat_name = k[len(clip_score_key) + 1:]
                    if img_obj[k] > csl_thres[cat_name]:  # background to similar to <cat_name>
                        is_valid_image = False
        if is_valid_image:
            new_images.append(img_obj)
            valid_image_ids.append(img_obj['id'])
            
    valid_ann_ids = coco.getAnnIds(imgIds=valid_image_ids)
    for anno in old_annos:
        if anno['id'] in valid_ann_ids:
            new_annos.append(anno)

    unfiltered_annotation["images"] = new_images
    unfiltered_annotation["annotations"] = new_annos

    with open(target_annotation_path, "w+") as f:
        json.dump(unfiltered_annotation, f)

    img_names = [img['file_name'] for img in unfiltered_annotation["images"]]
    for img_name in img_names:
        shutil.copy(os.path.join(unfiltered_image_folder, img_name), os.path.join(target_image_folder, img_name))

    print(f"num old_images: {len(old_images)}")
    print(f"num old_annos: {len(old_annos)}")
    print(f"num new_images: {len(new_images)}")
    print(f"num new_annos: {len(new_annos)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--unfiltered_data_folder", default=None, type=str)
    parser.add_argument("-k", "--clip_score_key", default=None, type=str)
    parser.add_argument("-p", "--percent_kept", default=None, type=int)
    args = parser.parse_args()

    '''
    e.g. coco10novel (coco 10 shot, novel cat only)
    python3 5c_filter_out_similar_background.py \
        -d /media/data/dad/cnet/experiments/coco10novel/mix_n2000_o1_s1_p640_pfa_csl_p30 \
        -k csl \
        -p 30
    '''

    filter_background(
            unfiltered_data_folder=args.unfiltered_data_folder,
            clip_score_key=args.clip_score_key,
            percent_kept=args.percent_kept,
        )


if __name__ == "__main__":
    main()
