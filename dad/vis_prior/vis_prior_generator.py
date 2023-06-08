import cv2 as cv
import numpy as np
import os

from edge_detector import CannyEdgeDetector, HEDEdgeDetector, MLSDEdgeDetector, MidasDepthDetector, UniformerMaskDetector


class VisPriorGenerator():
    def __init__(self, vpd, fill_val, detector, sample_channel=None):
        self.vpd = vpd
        self.detector = detector
        self.fill_val = fill_val
        self.sample_channel = sample_channel

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

    def draw_one_bbox(self, img, im_prior, new_bbox, fill_val=None, target_img=None):
        x_new, y_new, w_new, h_new = new_bbox
        H, W, _ = img.shape

        if fill_val is None:
            fill_val = self.fill_val
        
        # init target_img to draw if not provided
        if target_img is None:
            target_img = np.zeros((H,W,im_prior.shape[2]))
            target_img.fill(fill_val)
        else:
            assert target_img.shape[0] == H, f"shape miss match: {target_img.shape[0]} != {H}"
            assert target_img.shape[1] == W, f"shape miss match: {target_img.shape[1]} != {W}"
            assert target_img.shape[2] == sample_channel, f"shape miss match: {target_img.shape[2]} != {sample_channel}"

        im_prior = cv.resize(im_prior, dsize=(w_new, h_new))
        if im_prior.ndim == 2:  # cv resize will eliminate last dimension if it's 1
            im_prior = im_prior[:, :, None]

        target_img[int(y_new):int(y_new)+im_prior.shape[0], int(x_new):int(x_new)+im_prior.shape[1], :] = im_prior  # TODO: consider overwrite or add to previous bbox when we use multiple bbox in an image

        return target_img

    def visualize_one_bbox(self, img, bbox, category_name=None, prior_mode=None, fill_val=None, target_img=None):
        # used for simple visualization
        # img: source img

        new_bbox = self.vpd.generate_bbox(img, bbox)

        im_prior = self.detect_one_bbox(img, bbox)

        target_img = self.draw_one_bbox(img, im_prior, new_bbox, fill_val=fill_val, target_img=target_img)

        return target_img 


class CannyVPG(VisPriorGenerator):
    def __init__(self, vpd, fill_val=0, low=100, high=200, blur=5):
        super().__init__(vpd=vpd, fill_val=fill_val, detector=CannyEdgeDetector(low=low, high=high, blur=blur), sample_channel=1)
        self.low = low
        self.high = high
        self.blur = blur


class HEDVPG(VisPriorGenerator):
    def __init__(self, vpd, fill_val=0):
        super().__init__(vpd=vpd, fill_val=fill_val, detector=HEDEdgeDetector(), sample_channel=1)


class MLSDVPG(VisPriorGenerator):
    def __init__(self, vpd, fill_val=0, thr_v=0.1, thr_d=0.1):
        super().__init__(vpd=vpd, fill_val=fill_val, detector=MLSDEdgeDetector(thr_v=thr_v, thr_d=thr_d), sample_channel=1)


class MidasVPG(VisPriorGenerator):
    def __init__(self, vpd, fill_val=0, a=6.2):
        super().__init__(vpd=vpd, fill_val=fill_val, detector=MidasDepthDetector(a=a), sample_channel=1)


class UniformerVPG(VisPriorGenerator):
    def __init__(self, vpd, fill_val=0):
        super().__init__(vpd=vpd, fill_val=fill_val, detector=UniformerMaskDetector())
