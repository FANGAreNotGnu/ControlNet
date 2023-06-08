import cv2 as cv
from PIL import ImageFilter

from utils import *


PIL_FILTERS = {
    "BLUR": ImageFilter.BLUR,
    "CONTOUR": ImageFilter.CONTOUR, 
    "DETAIL": ImageFilter.DETAIL, 
    "EDGE_ENHANCE": ImageFilter.EDGE_ENHANCE, 
    "EDGE_ENHANCE_MORE": ImageFilter.EDGE_ENHANCE_MORE, 
    "EMBOSS": ImageFilter.EMBOSS, 
    "FIND_EDGES": ImageFilter.FIND_EDGES, 
    "SHARPEN": ImageFilter.SHARPEN, 
    "SMOOTH": ImageFilter.SMOOTH, 
    "SMOOTH_MORE": ImageFilter.SMOOTH_MORE,
}  # https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html

class PILEdgeDetector():
    def detect(self, im, modes):
        im1 = copy_cvbgr_to_pil(im)
        for mode in modes:
            if mode in PIL_FILTERS:
                im1 = im1.filter(PIL_FILTERS[mode])
            else:
                raise ValueError(f"{mode} is not supported as EdgeDetector's mode")
        return copy_pilrgb_to_cv(im1)

class CannyEdgeDetector():
    def detect(self, img, low=100, high=200, blur=5):
        print(img.shape)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(img, (blur, blur), 0)
        return cv.Canny(img, low, high)

# TODO: HED https://github.com/s9xie/hed
# TODO: MLSD https://github.com/navervision/mlsd
