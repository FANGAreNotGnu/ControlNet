import argparse
import glob
import json
import numpy as np
import os
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb


def mix_annotation(
            source_annotation_path,
            source_image_folder,
            syn_annotation_folder,
            target_folder=None,
        ):

    if target_folder is None:
        target_folder = syn_annotation_folder.replace("/syn_n", "/mix_n")
        print(f"target_folder not provided, using default target folder: {target_folder}")

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    target_image_folder = os.path.join(target_folder, "images")
    target_annotation_path = os.path.join(target_folder, "annotation.json")

    if not os.path.exists(target_image_folder):
        os.mkdir(target_image_folder)

    with open(source_annotation_path, "r") as f:
        source_annotation = json.load(f)
    coco = COCO(source_annotation_path)

    # 1. copy source image data
    img_names = [img['file_name'] for img in coco.loadImgs(coco.getImgIds())]
    for img_name in img_names:
        shutil.copy(os.path.join(source_image_folder, img_name), os.path.join(target_image_folder, img_name))

    # 2. add sync anno to annotation, copy syn image
    curr_img_id = max(coco.getImgIds()) + 1
    curr_ann_id = max(coco.getAnnIds()) + 1
    syn_data_paths = glob.glob(os.path.join(syn_annotation_folder, "*"))
    new_images = source_annotation["images"]
    new_anns = source_annotation["annotations"]

    for syn_data_path in tqdm(syn_data_paths):
        # copy sync image
        syn_image_path = os.path.join(syn_data_path, "syn000.jpg")  # source path  # TODO: only support sample = 1
        syn_image_name = "%012d.jpg"%curr_img_id  # target name
        shutil.copy(syn_image_path, os.path.join(target_image_folder, syn_image_name))
        
        layout_cats_path = os.path.join(syn_data_path, "layout_cats.npy")
        layout_bboxes_path = os.path.join(syn_data_path, "layout_bboxes.npy")
        prompt_path = os.path.join(syn_data_path, "prompt.npy")
        cats = np.load(layout_cats_path)
        bboxes = np.load(layout_bboxes_path)
        prompt = np.load(prompt_path)[0]
        
        catids = coco.getCatIds(catNms=cats)
        
        # add sync anno to annotation
        # TODO: image shape hard coded
        new_images.append({'file_name': syn_image_name, 'height': 640, 'width': 640, 'id': curr_img_id, 'prompt': prompt,})
        
        num_objects = cats.shape[0]
        for i in range(num_objects):
            new_anns.append({
                'image_id': curr_img_id, 
                'bbox': bboxes[i].tolist(), 
                'area': float(bboxes[i][-1] * bboxes[i][-2]), 
                'category_id': catids[i], 
                'id': curr_ann_id
            })
            curr_ann_id += 1
        
        curr_img_id += 1

    source_annotation["images"] = new_images
    source_annotation["annotations"] = new_anns

    with open(target_annotation_path, "w+") as f:
        json.dump(source_annotation, f)

    print(f"num images: {len(source_annotation['images'])}")
    print(f"num annotations: {len(source_annotation['annotations'])}")
    print(f"num categories: {len(source_annotation['categories'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--source_annotation_path", default=None, type=str)
    parser.add_argument("--source_image_folder", default="/media/data/coco17/coco/train2017", type=str)
    parser.add_argument("-s", "--syn_annotation_folder", default=None, type=str)
    parser.add_argument("--target_folder", default=None, type=str)
    args = parser.parse_args()

    '''
    e.g. coco10novel (coco 10 shot, novel cat only)
    python3 4_mix_annotation.py \
        -a /media/data/coco17/coco/seed1/10shot_novel.json \
        -s /media/data/dad/cnet/experiments/coco10novel/syn_n2000_o1_s1_p640
    '''

    mix_annotation(
            source_annotation_path=args.source_annotation_path,
            source_image_folder=args.source_image_folder,
            syn_annotation_folder=args.syn_annotation_folder,
            target_folder=args.target_folder,
        )


if __name__ == "__main__":
    main()

