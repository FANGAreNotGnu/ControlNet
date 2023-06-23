import argparse
import json
import os


# https://github.com/ZhangGongjie/Meta-DETR/blob/main/datasets/__init__.py
# Meta-settings for few-shot object detection: base / novel category split


FILTER_DICT = {
    "coco_base_class_ids": [
        8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90
    ],
    "coco_novel_class_ids": [
        1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72
    ],
    "voc_base1_class_ids": [
        1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20
    ],
    "voc_novel1_class_ids": [
        3, 6, 10, 14, 18
    ],
    "voc_base2_class_ids": [
        2, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20
    ],
    "voc_novel2_class_ids": [
        1, 5, 10, 13, 18
    ],
    "voc_base3_class_ids": [
        1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 19, 20
    ],
    "voc_novel3_class_ids": [
        4, 8, 14, 17, 18
    ],
}


def remove_novel_class(source_file_path, target_file_path, fileter_name):
    with open(source_file_path, "r") as f:
        gt = json.load(f)

    old_annos = gt["annotations"]
    old_images = gt["images"]
    print(f"#old annos: {len(old_annos)}")
    print(f"#old images: {len(old_images)}")
    new_annos = []
    new_images = []

    valid_image_id = set()
    for anno in old_annos:
        if anno["category_id"] in FILTER_DICT[fileter_name]:  # TODO: make it configurable
            new_annos.append(anno)
            valid_image_id.add(anno["image_id"])
    for image in old_images:
        if image["id"] in valid_image_id:  # TODO: make it configurable
            new_images.append(image)

    gt["annotations"] = new_annos
    gt["images"] = new_images
    print(f"#new annos: {len(new_annos)}")
    print(f"#new images: {len(new_images)}")
    
    with open(target_file_path, "w+") as f:
        json.dump(gt, f)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_file_path", default=None, type=str)
    parser.add_argument("-t", "--target_file_path", default=None, type=str)
    parser.add_argument("-f", "--fileter_name", default=None, type=str)

    '''
    e.g. coco-10shot-novelonly
    python3 0_remove_novel_class_from_coco_annotation.py \
        -s /media/data/coco17/coco/seed1/10shot.json \
        -t /media/data/coco17/coco/seed1/10shot_novel.json \
        -f coco_novel_class_ids
    '''

    args = parser.parse_args()

    remove_novel_class(source_file_path=args.source_file_path, target_file_path=args.target_file_path, fileter_name=args.fileter_name)


if __name__ == "__main__":
    main()
