import cv2 as cv
import numpy as np
from PIL import Image


def imread(im_path):
    return cv.imread(im_path)[:,:,::-1]


def imwrite(im_path, img):
    return cv.imwrite(im_path, img[:,:,::-1])


def copy_cvbgr_to_pil(img):
    im_np = img.copy()
    im_np = cv.cvtColor(im_np, cv.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


def copy_numpy_to_pil(img):
    im_np = img.copy()
    im_pil = Image.fromarray(img)
    return im_pil


def copy_pil_to_cv(img):
    im_pil = img.copy()
    im_np = np.asarray(im_pil)[:,:,::-1]
    return im_np


def copy_pil_to_numpy(img):
    im_pil = img.copy()
    im_np = np.asarray(im_pil)
    return im_np


def crop_bboxes(img, bboxes):
    # bboxes [[x1,y1,w,h],[x1,y1,w,h],...]
    # only keep bboxes areas of the image

    mask = np.zeros_like(img)
    for bbox in bboxes:
        x, y, w, h = bbox
        mask[int(y):int(y+h), int(x):int(x+w), :] = 1
    return img * mask
