import json
import numpy as np

from torch.utils.data import Dataset

from utils import imread


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

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)