import json
import numpy as np

from torch.utils.data import Dataset

import sys
sys.path.append("/home/ubuntu/dad/ControlNet/dad/vis_prior")
from utils import imread, resize


class VisPriorDataset(Dataset):
    def __init__(self, annotation_file):
        self.data = []
        with open(annotation_file, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filepath = item['source']
        target_filepath = item['target']
        prompt = item['prompt']

        source = imread(source_filepath)
        target = imread(target_filepath)

        # IMPORTANT!
        # Both image should be of the same size and resize to multiple of 64
        # now we resize but later we need to random crop! TODO!
        H, W, _ = source.shape
        new_H = H // 64 * 64
        new_W = W // 64 * 64

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        source = resize(img=source, new_W=new_W, new_H=new_H)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        target = resize(img=target, new_W=new_W, new_H=new_H)

        return dict(jpg=target, txt=prompt, hint=source)