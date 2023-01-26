import bz2
from glob import glob
import io
import pickle

import numpy as np
import torch

from pytorch3dunet.datasets.utils import ConfigDataset
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('SemanticMapDataset')

def get_bbox(m):
    idx = np.vstack(np.nonzero(m[0])) # channel 0 is the occupancy map
    min_ = np.min(idx, axis=1)
    max_ = np.max(idx, axis=1)
    return list(zip(min_, max_))

def mask_with_bbox(m, bbox):
    mask = np.zeros_like(m[0], dtype=bool)
    mask[bbox[0][0]:bbox[0][1],bbox[1][0]:bbox[1][1],bbox[2][0]:bbox[2][1]] = True
    return m*mask

# necessary to force unpickle on CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


class SemanticMapDataset(ConfigDataset):
    """
    Load compressed 3D semantic map pickles.
    """

    def __init__(self, dataset_path):
        self.filenames_ = []
        self.filenames_ = glob(f'{dataset_path}/*/*/map_pickles/*.pbz2')

    def __getitem__(self, index):
        fn = self.filenames_[index]
        data = CPU_Unpickler(bz2.open(fn, 'rb')).load() # dict
        
        pred_map = data['3d_map_cur'][0]
        gt_map = data['3d_map'][0]
        
        # remove irrelevant channels
        pred_map = pred_map[[0, *range(4,pred_map.shape[0])]]
        gt_map = gt_map[[0, *range(4,gt_map.shape[0])]]

        # TODO: crop gt_map to relevant area?

        return pred_map, gt_map

    def __len__(self):
        return len(self.filenames_)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        datasets = []
        for file_path in dataset_config[phase]['file_paths']:
            logger.info(f'Loading {phase} set from: {file_path}...')
            dataset = SemanticMapDataset(file_path)
            datasets.append(dataset)
        return datasets