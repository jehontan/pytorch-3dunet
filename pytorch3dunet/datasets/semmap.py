import bz2
from glob import glob
import io
import pickle
import os

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
        pred_paths = os.path.join(dataset_path, '*_learned_*', '*')
        gt_paths = os.path.join(dataset_path, '*_gt_*', '*')

        pred_filenames = glob(os.path.join(pred_paths, 'map_pickles', 'map_pickle_*.pbz2'))
        gt_filenames = glob(os.path.join(gt_paths, 'map_pickles', 'map_pickle_*.pbz2'))

        data = dict()

        # TODO: do the trajectories match?

        for fn in pred_filenames:
            key = os.path.join(*fn.split('/')[-3:])
            data[key] = {'pred': fn}

        for fn in gt_filenames:
            key = os.path.join(*fn.split('/')[-3:])
            data[key]['gt'] = fn

        self.filenames = list(data.values())

    def __getitem__(self, index):
        pred_fn = self.filenames[index]['pred']
        gt_fn = self.filenames[index]['gt']
        
        pred_data = CPU_Unpickler(bz2.open(pred_fn, 'rb')).load() # dict
        gt_data = CPU_Unpickler(bz2.open(gt_fn, 'rb')).load() # dict

        pred_map = pred_data['3d_map'][0]
        gt_map = gt_data['3d_map'][0]
        
        # remove irrelevant channels
        pred_map = pred_map[[0, *range(4,pred_map.shape[0])]]
        gt_map = gt_map[[0, *range(4,gt_map.shape[0])]]

        # TODO: crop gt_map to relevant area?

        return pred_map, gt_map

    def __len__(self):
        return len(self.filenames)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        datasets = []
        for file_path in dataset_config[phase]['file_paths']:
            logger.info(f'Loading {phase} set from: {file_path}...')
            dataset = SemanticMapDataset(file_path)
            datasets.append(dataset)
        return datasets