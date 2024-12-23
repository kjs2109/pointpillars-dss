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
    from model.anchors import Anchors, anchor_target 
    import matplotlib.pyplot as plt


    def vis_numpy_array(np_array, pause_time=None):

        if np_array.__class__ == str:
            np_array = np.load(np_array)

        if len(np_array.shape) == 4: 
            np_array = np_array.squeeze(0) 

        if (np_array.shape[-1] != 3) and (len(np_array.shape) == 3): 
            np_array = np_array.transpose((1, 2, 0)) 

        # 시각화
        plt.figure(figsize=(10, 8))
        plt.imshow(np_array, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Value')  

        # 그리드 추가
        plt.grid(color='white', linestyle='--', linewidth=0.5)

        # 그리드 맞추기
        plt.xticks(np.arange(-0.5, np_array.shape[1], 1), labels=[])
        plt.yticks(np.arange(-0.5, np_array.shape[0], 1), labels=[])
        plt.gca().set_xticks(np.arange(-0.5, np_array.shape[1], 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, np_array.shape[0], 1), minor=True)
        plt.gca().grid(which='minor', color='white', linestyle='--', linewidth=0.5)

        plt.title("Grid Visualization of Output Array")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        if pause_time is not None: 
            plt.pause(pause_time) 
            plt.close() 

        plt.show() 


    assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]

    # anchors
    ranges = [[-40, -40, -3, 40, 40, 1], 
                [-40, -40, -3, 40, 40, 1], 
                [-40, -40, -3, 40, 40, 1]] 
    sizes = [[1.6, 3.9, 1.56], [0.6, 0.8, 1.73], [0.6, 1.76, 1.73]] 
    rotations=[0, 1.57]

    anchors_generator = Anchors(ranges=ranges, 
                                        sizes=sizes, 
                                        rotations=rotations)

    anchor = anchors_generator.get_multi_anchors(torch.tensor([248, 216]))



    dss_dataset = DssDataset(data_root='/workspace/DssDataset', split='train')
    print(len(dss_dataset))
    data = dss_dataset.__getitem__(1) 

    batched_anchors = torch.tensor(anchor).unsqueeze(0) 
    batched_gt_bboxes = torch.tensor(data['gt_bboxes_3d']).unsqueeze(0) 
    batched_gt_labels = torch.tensor(data['gt_labels']).unsqueeze(0) 


    anchor_target_dict = anchor_target(batched_anchors=batched_anchors, 
                                               batched_gt_bboxes=batched_gt_bboxes, 
                                               batched_gt_labels=batched_gt_labels, 
                                               assigners=assigners,
                                               nclasses=3) 


    print(anchor_target_dict.keys()) 
    print(anchor_target_dict)

    # dict_keys(['batched_labels', 'batched_label_weights', 'batched_bbox_reg', 'batched_bbox_reg_weights', 'batched_dir_labels', 'batched_dir_labels_weights'])

    print(anchor_target_dict['batched_bbox_reg'].shape)
    print(torch.max(anchor_target_dict['batched_bbox_reg'], dim=1)) 
    vis_numpy_array(anchor_target_dict['batched_bbox_reg'][0][247741-10:247741+10].detach().numpy()) 

