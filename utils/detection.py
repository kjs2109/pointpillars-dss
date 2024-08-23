import sys 
import os 
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))
from model import PointPillars 

import torch  


def inference(pc, ckpt_path): 
    # CLASSES = {
    #     'Unknown': 0,
    # } 
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
    }

    LABEL2CLASSES = {v:k for k, v in CLASSES.items()} 

    # load model 
    model = PointPillars(nclasses=len(CLASSES)).cuda()  
    model.load_state_dict(torch.load(ckpt_path)) 

    # load point cloud (input data) 
    pc_torch = torch.from_numpy(pc).cuda() 

    model.eval() 
    with torch.no_grad():
        output = model(batched_pts=[pc_torch], mode='test')[0]

    return output


def inference_from_model(batched_pts, model, mode='test'): 
    CLASSES = {
        'Unknown': 0, 
        'Unknown1': 1, 
        'Unknown2': 2,
    } 
    # CLASSES = {
    #     'Pedestrian': 0, 
    #     'Cyclist': 1, 
    #     'Car': 2
    # }


    model.eval() 
    with torch.no_grad(): 
        batched_output = model(batched_pts=batched_pts, mode=mode)

    return batched_output