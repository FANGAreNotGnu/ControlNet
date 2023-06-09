import cv2 as cv
import numpy as np
import os
from collections import defaultdict

from utils import imread
from vis_prior_detector import CannyEdgeDetector, HEDEdgeDetector, MLSDEdgeDetector, MidasDepthDetector, UniformerMaskDetector


class VisPriorGenerator():
    def __init__(self, vpl, fill_val, detector, sample_channel=None, annotation=None, im_folder=None):
        self.vpl = vpl
        self.detector = detector
        self.fill_val = fill_val
        self.sample_channel = sample_channel

        self.prior_bank = defaultdict(list)
        self.annotation = annotation
        self.im_folder = im_folder

        if self.annotation is not None and im_folder is not None:
            self.update_prior_bank(self.annotation, self.im_folder)

    def update_prior_bank(self, annotation, im_folder):
        if isinstance(annotation, str):
            with open(annotation, "r") as f:
                annos = json.load(f)
        elif isinstance(annotation, dict):
            annos = annotation
        else:
            raise ValueError(f"annotation should be a str (json's path) or a dict, but it is {type(annotation)}")

        for ann in annos["annotations"]:
            image_id = ann["image_id"]
            image_object = annos["images"][image_id]
            assert image_object["id"] == image_id, f"image id not equals to image idx, image idx: {image_id}, image_id: {image_object['id']}"

            category_id = ann["category_id"]
            category_object = annos["categories"][category_id]
            assert category_object["id"] == category_id, f"category id not equals to category idx, category idx: {category_id}, category_id: {category_object['id']}"

            image_file_name = image_object['file_name']
            category_name = category_object['name']

            img = imread(os.path.join(im_folder, image_file_name))
            bbox = ann_id['bbox']

            vis_prior = self.detect_one_bbox(img, bbox)

            self.prior_bank[category_name].append(vis_prior)

        print("prior bank updated")

    def generate_one_dataset(
        data_path,
        labels_path=None,
    ):
        if not labels_path:
            labels_path = os.path.join(data_path, "_annotations.coco.json")

        raise NotImplementedError

    def detect_one_bbox(self, img, bbox):
        x, y, w, h = bbox

        im_cropped = img[int(y):int(y+h), int(x):int(x+w), :].copy()
        im_prior = self.detector(img=im_cropped)
        if im_prior.ndim == 2:
            im_prior = im_prior[:, :, None]

        return im_prior

    def draw_one_bbox(self, im_prior, new_bbox, fill_val=None, target_img=None, im_shape=None):
        x_new, y_new, w_new, h_new = new_bbox

        if fill_val is None:
            fill_val = self.fill_val
        
        # init target_img to draw if not provided
        if target_img is None:
            assert im_shape is not None, "Failed to initialize the target image! Note that target_img and im_shape can not both be None."
            target_img = np.zeros((im_shape[0],im_shape[1],im_prior.shape[2]))
            target_img.fill(fill_val)
        else:
            assert target_img.shape[0] == im_shape[0], f"shape miss match: {target_img.shape[0]} != {im_shape[0]}"
            assert target_img.shape[1] == im_shape[1], f"shape miss match: {target_img.shape[1]} != {im_shape[1]}"
            assert target_img.shape[2] == sample_channel, f"shape miss match: {target_img.shape[2]} != {sample_channel}"

        im_prior = cv.resize(im_prior, dsize=(w_new, h_new))
        if im_prior.ndim == 2:  # cv resize will eliminate last dimension if it's 1
            im_prior = im_prior[:, :, None]

        target_img[int(y_new):int(y_new)+im_prior.shape[0], int(x_new):int(x_new)+im_prior.shape[1], :] = im_prior  # TODO: consider add (with transparency?) to previous bbox when we use multiple bbox in an image (currently overwrite)

        return target_img

    def draw_one_layout(self, im_shape, layout, fill_val=None):
        # layout (without prior): [[category1, bbox1], [category2, bbox2], ...]
        # layout (with prior): [[category1, bbox1, prior1], [category2, bbox2, prior2], ...]
        # layout is ordered
        # im_shape: H, W, C

        if fill_val is None:
            fill_val = self.fill_val

        target_img = np.zeros(im_shape)
        target_img.fill(fill_val)

        for layout_object in layout:
            if len(layout_object) == 2:
                category_name, bbox = layout_object
                im_prior = self.sample_im_prior(category_name)  #TODO
            elif len(layout_object) == 3:
                category_name, bbox, im_prior = layout_object
            else:
                raise ValueError(f"the length of layout_object should be 2 or 3, but is {len(layout_object)}")

            target_img = draw_one_bbox(self, im_prior=im_prior, new_bbox=bbox, fill_val=fill_val, target_img=target_img, im_shape=im_shape)

        raise target_img

    def visualize_one_bbox(self, img, bbox, category_name=None, prior_mode=None, fill_val=None, target_img=None):
        # used for simple visualization
        # img: source img

        new_bbox = self.vpl.generate_bbox(im_shape=img.shape, prior_shape=(bbox[3],bbox[2]))  # prior shape=hw

        im_prior = self.detect_one_bbox(img, bbox)

        target_img = self.draw_one_bbox(im_prior, new_bbox, fill_val=fill_val, target_img=target_img, im_shape=img.shape if target_img is None else target_img.shape)

        return target_img 


class CannyVPG(VisPriorGenerator):
    def __init__(self, vpl, fill_val=0, low=100, high=200, blur=5):
        super().__init__(vpl=vpl, fill_val=fill_val, detector=CannyEdgeDetector(low=low, high=high, blur=blur), sample_channel=1)
        self.low = low
        self.high = high
        self.blur = blur


class HEDVPG(VisPriorGenerator):
    def __init__(self, vpl, fill_val=0):
        super().__init__(vpl=vpl, fill_val=fill_val, detector=HEDEdgeDetector(), sample_channel=1)


class MLSDVPG(VisPriorGenerator):
    def __init__(self, vpl, fill_val=0, thr_v=0.1, thr_d=0.1):
        super().__init__(vpl=vpl, fill_val=fill_val, detector=MLSDEdgeDetector(thr_v=thr_v, thr_d=thr_d), sample_channel=1)


class MidasVPG(VisPriorGenerator):
    def __init__(self, vpl, fill_val=0, a=6.2):
        super().__init__(vpl=vpl, fill_val=fill_val, detector=MidasDepthDetector(a=a), sample_channel=1)


class UniformerVPG(VisPriorGenerator):
    def __init__(self, vpl, fill_val=0):
        super().__init__(vpl=vpl, fill_val=fill_val, detector=UniformerMaskDetector())
