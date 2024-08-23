import argparse
import numpy as np
import os
import torch
import pdb
from tqdm import tqdm

from utils import write_pickle, inference_from_model, iou3d  # write_label,iou3d
from dataset import DssDataset, get_dataloader
from model import PointPillars


def write_label(result, file_path, suffix='.txt'):
    '''
    result: dict,
    file_path: str
    '''
    assert os.path.splitext(file_path)[1] == suffix
    name, truncated, occluded, dimensions, location, rotation_y, score = result['name'], result['truncated'], result['occluded'], \
                                                                         result['dimensions'], result['location'], result['rotation_y'], \
                                                                         result['score']
    
    with open(file_path, 'w') as f:
        for i in range(len(name)):
            hwl = ' '.join(map(str, dimensions[i]))
            xyz = ' '.join(map(str, location[i]))
            line = f'{name[i]} {truncated[i]} {occluded[i]} {hwl} {xyz} {rotation_y[i]} {score[i]}\n'
            f.writelines(line)

def keep_bbox_from_lidar_range(result, pcd_limit_range):
    '''
    result: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    pcd_limit_range: []
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    '''
    lidar_bboxes, labels, scores = result['lidar_bboxes'], result['labels'], result['scores']
    if 'bboxes2d' not in result:
        result['bboxes2d'] = np.zeros_like(lidar_bboxes[:, :4])
    if 'camera_bboxes' not in result:
        result['camera_bboxes'] = np.zeros_like(lidar_bboxes)
    bboxes2d, camera_bboxes = result['bboxes2d'], result['camera_bboxes']
    flag1 = lidar_bboxes[:, :3] > pcd_limit_range[:3][None, :] # (n, 3)
    flag2 = lidar_bboxes[:, :3] < pcd_limit_range[3:][None, :] # (n, 3)
    keep_flag = np.all(flag1, axis=-1) & np.all(flag2, axis=-1)
    
    result = {
        'lidar_bboxes': lidar_bboxes[keep_flag],
        'labels': labels[keep_flag],
        'scores': scores[keep_flag],
        'bboxes2d': bboxes2d[keep_flag],
        'camera_bboxes': camera_bboxes[keep_flag]
    }
    
    return result

def get_score_thresholds(tp_scores, total_num_valid_gt, num_sample_pts=41):
    score_thresholds = []
    tp_scores = sorted(tp_scores)[::-1]
    cur_recall, pts_ind = 0, 0
    for i, score in enumerate(tp_scores):
        lrecall = (i + 1) / total_num_valid_gt
        rrecall = (i + 2) / total_num_valid_gt

        if i == len(tp_scores) - 1:
            score_thresholds.append(score)
            break

        if (lrecall + rrecall) / 2 < cur_recall:
            continue

        score_thresholds.append(score)
        pts_ind += 1
        cur_recall = pts_ind / (num_sample_pts - 1)
    return score_thresholds 

def do_eval(det_results, gt_results, CLASSES, saved_path):  
    '''
    det_results: list , 
    gt_results: dict(id -> det_results) 
    CLASSES: dict 
    '''
    assert len(det_results) == len(gt_results) 
    f = open(os.path.join(saved_path, 'eval_results.txt'), 'w') 

    # 1. calculate iou 
    ious = {
        'bbox_2d': [],
        'bbox_dev': [],  
        'bbox_3d': [],
    } 
    ids = list(sorted(gt_results.keys())) 
    for id in ids: 
        gt_result = gt_results[id]['annos'] 
        det_result = det_results[id] 

        det_location = det_result['location'].astype(np.float32) 
        det_dimensions = det_result['dimensions'].astype(np.float32) 
        det_rotation_y = det_result['rotation_y'].astype(np.float32) 

        gt_bboxes3d = np.array(gt_result['label']) 
        det_bboxes3d = np.concatenate([det_location, det_dimensions, det_rotation_y[:, None]], axis=-1) 

        iou3d_v = iou3d(torch.from_numpy(gt_bboxes3d).cuda(), torch.from_numpy(det_bboxes3d).cuda()) 
        ious['bbox_3d'].append(iou3d_v.cpu().numpy()) 

    # print(len(ious['bbox_3d'])) 
    # print(len(ious['bbox_2d']))
    # print(ious['bbox_3d'][0].shape) 
    # print(ious['bbox_3d'][1].shape) 
    # print(ious['bbox_3d'][2].shape)

    MIN_IOUS = {
        'Unknown': [0.5, 0.5, 0.5], 
        'Unknown1': [0.5, 0.5, 0.5], 
        'Unknown2': [0.7, 0.7, 0.7],
    }

    MIN_HEIGHTS = [40, 25, 25] 

    overall_results = {}  #################################################### 
    for e_ind, eval_type in enumerate(['bbox_3d']): 
        eval_ious = ious[eval_type] 
        eval_ap_results = {}
        for cls in CLASSES: 
            eval_ap_results[cls] = [] 
            CLS_MIN_IOU = MIN_IOUS[cls][e_ind] 

            for difficulty in [0, 1, 2]:
                # 1. bbox property
                total_gt_ignores, total_det_ignores, total_dc_bboxes, total_scores = [], [], [], []
                # total_gt_alpha, total_det_alpha = [], [] 
                for id in ids: 
                    # 위 리스트 적절한 값 채우기 
                    gt_result = gt_results[id]['annos'] 
                    det_result = det_results[id] 

                    # 1.1 gt bbox property 
                    cur_gt_names = gt_result['name']  
                    cur_difficulty = [1] * len(cur_gt_names) # 0: easy, 1: moderate, 2: hard 
                    gt_ignores, dc_bboxes = [], [] 
                    for j, cur_gt_name in enumerate(cur_gt_names): 
                        ignore = cur_difficulty[j] < 0 or cur_difficulty[j] > difficulty  

                        if cur_gt_name == cls:
                            valid_class = 1
                        elif cls == 'Pedestrian' and cur_gt_name == 'Person_sitting':
                            valid_class = 0
                        elif cls == 'Car' and cur_gt_name == 'Van':
                            valid_class = 0
                        else:
                            valid_class = -1 

                        if valid_class == 1 and not ignore: 
                            gt_ignores.append(0) 
                        elif valid_class == 0 and not ignore: 
                            gt_ignores.append(1) 
                        else: 
                            gt_ignores.append(-1) 

                        if cur_gt_name == 'DontCare': 
                            dc_bboxes.append(gt_result['bbox'][j]) 

                    total_gt_ignores.append(gt_ignores) 
                    total_dc_bboxes.append(np.array(dc_bboxes))  

                    # 1.2 det bbox property 
                    cur_det_names = det_result['name'] 
                    cur_det_heights = [50] * len(cur_det_names)  # det_result['bbox'][:, 3] - det_result['bbox'][:, 1]  # 2d bbox height 
                    det_ignores = [] 
                    for j, cur_det_name in enumerate(cur_det_names): 
                        if cur_det_heights[j] < MIN_HEIGHTS[difficulty]:  
                            det_ignores.append(1) 
                        elif cur_det_name == cls: 
                            det_ignores.append(0) 
                        else: 
                            det_ignores.append(-1) 
                    total_det_ignores.append(det_ignores) 
                    total_scores.append(det_result['score']) 


                # 2. calculate scores thresholds for PR curve 
                tp_scores = [] 
                for i, id in enumerate(ids):
                    cur_eval_ious = eval_ious[i]
                    gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                    scores = total_scores[i]

                    nn, mm = cur_eval_ious.shape
                    assigned = np.zeros((mm, ), dtype=np.bool_)
                    for j in range(nn):
                        if gt_ignores[j] == -1:
                            continue
                        match_id, match_score = -1, -1
                        for k in range(mm):
                            if not assigned[k] and det_ignores[k] >= 0 and cur_eval_ious[j, k] > CLS_MIN_IOU and scores[k] > match_score:
                                match_id = k
                                match_score = scores[k]
                        if match_id != -1:
                            assigned[match_id] = True
                            if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                tp_scores.append(match_score)
                total_num_valid_gt = np.sum([np.sum(np.array(gt_ignores) == 0) for gt_ignores in total_gt_ignores])
                score_thresholds = get_score_thresholds(tp_scores, total_num_valid_gt) 


                # 3. draw PR curve and calculate mAP 
                tps, fns, fps = [], [], []

                for score_threshold in score_thresholds:
                    tp, fn, fp = 0, 0, 0

                    for i, id in enumerate(ids):
                        cur_eval_ious = eval_ious[i]
                        gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                        # gt_alpha, det_alpha = total_gt_alpha[i], total_det_alpha[i]
                        scores = total_scores[i]

                        nn, mm = cur_eval_ious.shape
                        assigned = np.zeros((mm, ), dtype=np.bool_)
                        for j in range(nn):
                            if gt_ignores[j] == -1:
                                continue
                            match_id, match_iou = -1, -1
                            for k in range(mm):
                                if not assigned[k] and det_ignores[k] >= 0 and scores[k] >= score_threshold and cur_eval_ious[j, k] > CLS_MIN_IOU:
    
                                    if det_ignores[k] == 0 and cur_eval_ious[j, k] > match_iou:
                                        match_iou = cur_eval_ious[j, k]
                                        match_id = k
                                    elif det_ignores[k] == 1 and match_iou == -1:
                                        match_id = k

                            if match_id != -1:
                                assigned[match_id] = True
                                if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                    tp += 1

                            else:
                                if gt_ignores[j] == 0:
                                    fn += 1
                            
                        for k in range(mm):
                            if det_ignores[k] == 0 and scores[k] >= score_threshold and not assigned[k]:
                                fp += 1

                    tps.append(tp) 
                    fns.append(fn) 
                    fps.append(fp) 
                    
                tps, fns, fps = np.array(tps), np.array(fns), np.array(fps) 

                recalls = tps / (tps + fns) 
                precisions = tps / (tps + fps)
                for i in range(len(score_thresholds)): 
                    precisions[i] = np.max(precisions[i:]) 

                sums_AP = 0 
                for i in range(0, len(score_thresholds), 4): 
                    sums_AP += precisions[i] 

                mAP = sums_AP / 11 * 100  #  recall 0.0 ~ 1.0 (11 points) 
                eval_ap_results[cls].append(mAP) 

        print(f'=========={eval_type.upper()}==========')
        print(f'=========={eval_type.upper()}==========', file=f)
        for k, v in eval_ap_results.items():
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)

        overall_results[eval_type] = np.mean(list(eval_ap_results.values()), 0)
    
    print(f'\n==========Overall==========')
    print(f'\n==========Overall==========', file=f)
    for k, v in overall_results.items():
        print(f'{k} AP: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
        print(f'{k} AP: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)
    f.close()
                        




def main(args): 
    val_dataset = DssDataset(data_root=args.data_root, split='val')
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)
    CLASSES = DssDataset.CLASSES
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}

    if not args.no_cuda:
        model = PointPillars(nclasses=args.nclasses).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=args.nclasses)
        model.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu')))
    
    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)
    saved_submit_path = os.path.join(saved_path, 'submit')
    os.makedirs(saved_submit_path, exist_ok=True)

    pcd_limit_range = np.array([-40, -40, -3, 40, 40, 1], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        format_results = {}
        print('Predicting and Formatting the results.')
        for i, data_dict in enumerate(val_dataloader):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batched_difficulty = data_dict['batched_difficulty']
            batch_results = inference_from_model(batched_pts, model, mode='val') 

            for j, result in enumerate(batch_results):
                format_result = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }
                
                # calib_info = data_dict['batched_calib_info'][j]
                # tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
                # r0_rect = calib_info['R0_rect'].astype(np.float32)
                # P2 = calib_info['P2'].astype(np.float32)
                # image_shape = data_dict['batched_img_info'][j]['image_shape']
                idx = data_dict['batched_img_info'][j]['image_idx']
                result_filter = keep_bbox_from_lidar_range(result, pcd_limit_range)

                # batch output check 
                lidar_bboxes = result_filter['lidar_bboxes']
                labels, scores = result_filter['labels'], result_filter['scores']

                for lidar_bbox, label, score in zip(lidar_bboxes, labels, scores):
                    format_result['name'].append(LABEL2CLASSES[label])
                    format_result['truncated'].append(0.0)
                    format_result['occluded'].append(0)
                    format_result['dimensions'].append(lidar_bbox[3:6])
                    format_result['location'].append(lidar_bbox[:3])
                    format_result['rotation_y'].append(lidar_bbox[6])
                    format_result['score'].append(score)
                
                write_label(format_result, os.path.join(saved_submit_path, f'{idx:06d}.txt'))

                format_results[idx] = {k:np.array(v) for k, v in format_result.items()} 

                print(f'Index: {idx}, class: {format_result["name"]}, score: {format_result["score"]}')
        
        write_pickle(format_results, os.path.join(saved_path, 'results.pkl'))
    
    print('Evaluating.. Please wait several seconds.')
    do_eval(format_results, val_dataset.data_infos, CLASSES, saved_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/workspace/DssDataset', help='your data root for dss')
    parser.add_argument('--ckpt', default='./demo/checkpoints/pointpillars_159.pth', help='your checkpoint for dss')
    parser.add_argument('--saved_path', default='results', help='your saved path for predicted results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--no_cuda', action='store_true', help='whether to use cuda')
    args = parser.parse_args()

    main(args)