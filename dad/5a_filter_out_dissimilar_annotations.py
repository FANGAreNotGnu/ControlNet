from collections import defaultdict
import argparse
import glob
import json
import numpy as np
import os
import shutil
from tqdm import tqdm


def filter_annotation(
    unfiltered_data_folder,
    clip_score_key,
    percent_kept,
):
    unfiltered_annotation_path = os.path.join(unfiltered_data_folder, "annotation.json")
    unfiltered_image_folder = os.path.join(unfiltered_data_folder, "images")
    target_folder = unfiltered_data_folder + "_pfa_%s%d" % (clip_score_key, percent_kept)

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    target_image_folder = os.path.join(target_folder, "images")
    target_annotation_path = os.path.join(target_folder, "annotation.json")

    if not os.path.exists(target_image_folder):
        os.mkdir(target_image_folder)

    with open(unfiltered_annotation_path, "r") as f:
        unfiltered_annotation = json.load(f)


    cat_ids = [cat['id'] for cat in unfiltered_annotation["categories"]]

    clip_scores = defaultdict(list)
    for ann in unfiltered_annotation["annotations"]:
        if "segmentation" not in ann:  # only use synthetic data (no GT data) to set the threshold
            clip_scores[ann['category_id']].append(ann[clip_score_key])

    csl_thres = {category_id:np.percentile(scores, 100 - percent_kept) for category_id, scores in clip_scores.items()}

    old_images = unfiltered_annotation["images"]
    old_annos = unfiltered_annotation["annotations"]
    new_annos = []
    new_images = []

    valid_image_id = set()
    for anno in old_annos:
        if "segmentation" in anno or anno[clip_score_key] > csl_thres[anno["category_id"]]:  # if segmentation in ann it's gt TODO: add sync flag
            new_annos.append(anno)
            valid_image_id.add(anno["image_id"])
    for image in old_images:
        if image["id"] in valid_image_id:
            new_images.append(image)

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
    python3 5a_filter_out_dissimilar_annotations.py \
        -d /media/data/dad/cnet/experiments/coco10novel/mix_n2000_o1_s1_p640 \
        -k csl_p \
        -p 20
    '''

    filter_annotation(
            unfiltered_data_folder=args.unfiltered_data_folder,
            clip_score_key=args.clip_score_key,
            percent_kept=args.percent_kept,
        )


if __name__ == "__main__":
    main()
