import numpy as np 
import os 
import torch 
from torch.utils.data import Dataset 

import sys 
BASE = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(BASE)) 

from utils import read_pickle, read_ply 


class DssDataset(Dataset): 

    CLASSES = {
        'Unknown': 0, 
        'Unknown1': 1, 
        'Unknown2': 2,
    }

    def __init__(self, data_root, split): 
        self.data_root = data_root 
        self.split = split 
        self.data_infos = read_pickle(os.path.join(data_root, f'dss_infos_{split}.pkl')) 
        self.sorted_ids = list(self.data_infos.keys()) 

    def __len__(self): 
        return len(self.data_infos)
    
    def __getitem__(self, index): 
        data_info = self.data_infos[self.sorted_ids[index]] 
        image_info, annos_info = data_info['image'], data_info['annos'] 

        pcd_path = data_info['pcd_path'] 
        pts = read_ply(pcd_path)

        annos_name = annos_info['name'] 
        gt_labels = [self.CLASSES[name] for name in annos_name]   
        gt_bboxes = annos_info['label'] 

        data_dict = {
            'pts': pts, 
            'gt_bboxes_3d': np.array(gt_bboxes), 
            'gt_labels': np.array(gt_labels), 
            'gt_names': annos_name, 
            'difficulty': annos_info['difficulty'], 
            'image_info': image_info, 
            'calib_info': None, 
        }    

        return data_dict 
    

if __name__ == '__main__': 

    dss_dataset = DssDataset(data_root='/workspace/DssDataset', split='val')
    print(len(dss_dataset))
    print(dss_dataset.__getitem__(1))