import argparse
import glob
import json
import os
from tqdm import tqdm

from pycocotools.coco import COCO  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb

from vis_prior.utils import imread, imwrite
from vis_prior.vis_prior_generator import CannyVPG, HEDVPG, MLSDVPG, MidasVPG, UniformerVPG
from vis_prior.vis_prior_layout import UniformRandomNoClipVPL


def generate_vis_prior(source_image_folder, source_annotation_file, output_parent_folder, vis_prior_name, ext):
    output_folder = os.path.join(output_parent_folder, vis_prior_name)
    output_im_prior_folder = os.path.join(output_folder, "vis_prior_images")
    output_annotation_path = os.path.join(output_folder, "vis_prior_annotation.json")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(output_im_prior_folder):
        os.mkdir(output_im_prior_folder)

    coco = COCO(source_annotation_file)
    naive_vpd = UniformRandomNoClipVPL()
    vpg = HEDVPG(vpl=naive_vpd, fill_val=0, annotation=None, im_folder=None)

    img_ids = coco.getImgIds()

    with open(output_annotation_path, "w+") as f:
        for img_id in tqdm(img_ids):
            coco_img = coco.loadImgs(img_id)[0]
            source_img_name = coco_img['file_name']
            
            source_img_path = os.path.join(source_image_folder, source_img_name)
            im_prior_path = os.path.join(output_im_prior_folder, source_img_name)
            
            source_img = imread(source_img_path)
            annIds = coco.getAnnIds(imgIds=coco_img['id'])
            anns = coco.loadAnns(annIds)
            cat_names = [cat['name'] for cat in coco.loadCats([ann['category_id'] for ann in anns])]
            bboxes = [ann['bbox'] for ann in anns]

            prompt = ', '.join(cat_names)  # TODO: we can use set to remove duplicates but shall we do that?
            
            im_prior = vpg.detect_one_img(img=source_img,bboxes=bboxes)
            imwrite(im_prior_path, im_prior)
            
            json_row = json.dumps({"source":im_prior_path,"target":source_img_path,"prompt": prompt})
            f.write(json_row + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--source_image_folder", default=None, type=str)
    parser.add_argument("-a", "--source_annotation_file", default=None, type=str)
    parser.add_argument("-o", "--output_parent_folder", default="/media/data/dad/cnet/vispriors", type=str)
    parser.add_argument("-v", "--vis_prior_name", default=None, type=str)
    parser.add_argument("--ext", default="jpg", type=str)
    args = parser.parse_args()


    '''
    e.g. coco-10shot-novelonly
    python3 1_generate_vis_priors_for_controlnet_finetuning.py \
        -i /media/data/coco17/coco/train2017/ \
        -a /media/data/coco17/coco/seed1/10shot_novel.json \
        -v coco10novel
    '''


    generate_vis_prior(
            source_image_folder=args.source_image_folder,
            source_annotation_file=args.source_annotation_file,
            output_parent_folder=args.output_parent_folder,
            vis_prior_name=args.vis_prior_name,
            ext=args.ext,
        )


if __name__ == "__main__":
    main()