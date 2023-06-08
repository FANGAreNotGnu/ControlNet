import numpy as np
import os

from edge_detector import CannyEdgeDetector

class VisPriorGenerator():
    def __init__(self, vpd):
        self.vpd = vpd

    def generate_one_dataset(
        data_path,
        labels_path=None,
    ):
        if not labels_path:
            labels_path = os.path.join(data_path, "_annotations.coco.json")

        raise NotImplementedError

    def generate_one_sample(img, category_name, bbox, prior_mode):
        pass

class CannyVisPriorGenerator(VisPriorGenerator):
    def __init__(self, vpd):
        self.vpd = vpd
        self.edge_detector = CannyEdgeDetector()

    def generate_one_sample(self, img, bbox, category_name, prior_mode, fill_val=0, low=100, high=200, blur=5):
        x, y, w, h = bbox
        H, W, _ = img.shape
        x_new, y_new, w_new, h_new = self.vpd.generate_bbox(W=W, H=H, x=x, y=y, w=w, h=h)

        im_cropped = img[int(y):int(y+h), int(x):int(x+w), :].copy()
        im_edge = self.edge_detector.detect(img=im_cropped,low=low,high=high)

        out_im = np.zeros((H, W))  # gray scale for canny prior
        out_im.fill(fill_val)  # gray scale for canny prior

        out_im[int(y_new):int(y_new)+im_edge.shape[0], int(x_new):int(x_new)+im_edge.shape[1]] = im_edge

        return out_im 
