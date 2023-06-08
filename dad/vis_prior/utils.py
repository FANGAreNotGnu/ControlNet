import cv2 as cv
import numpy as np
from PIL import Image

def imread(im_path):
    return cv.imread(im_path)

def copy_cvbgr_to_pil(img):
    im_np = img.copy()
    im_np = cv.cvtColor(im_np, cv.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

def copy_pilrgb_to_cv(img):
    im_pil = img.copy()
    im_np = np.asarray(im_pil)[:,:,::-1]
    return im_np
