import argparse
import pdb
import cv2
import numpy as np
import pandas as pd 
import os
from tqdm import tqdm
import sys
import json 
import math 
CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR) 

from utils import write_pickle 


def default_data_split(data_root): 
    scenarios = sorted([scenario for scenario in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, scenario))]) 

    scenario_count = len(scenarios) 
    train_index = int(scenario_count * 0.8) 
    val_index = int(scenario_count * 0.1) + train_index

    split_data = {
        'train': scenarios[:train_index], 
        'val': scenarios[train_index:val_index], 
        'test': scenarios[val_index:], 
    }
    
    return split_data 


def create_dataframe(folder_structer): 

    root_dir = folder_structer['root_dir'] 
    raw_data = folder_structer['raw_data']
    label_data = folder_structer['label_data']
    raw_data_type = folder_structer['raw_data_type']
    sub_data_type = folder_structer['sub_data_type']
    label_type = folder_structer['label_type']
    split_info = folder_structer['split'] if folder_structer['split'] != {} else default_data_split(os.path.join(root_dir, raw_data)) 
    print(split_info)
    scenarios = [] 
    curr_scenario = []

    pcd_fnames = [] 
    pcd_fpaths = []
    image_fnames = [] 
    image_fpaths = []  
    label_fnames = [] 
    label_fpaths = [] 
    split = []

    for scenario in os.listdir(os.path.join(root_dir, raw_data)): 
        scenarios.append(scenario) 

        if split_info['train'] and scenario in split_info['train']: 
            split_type = 'train' 
        elif split_info['val'] and scenario in split_info['val']: 
            split_type = 'val' 
        elif split_info['test'] and scenario in split_info['test']: 
            split_type = 'test' 
        else: 
            raise ValueError('please check the split info')

        print(f'start {scenario}')
        # find pcd & image files and label files 
        for design_id in os.listdir(os.path.join(root_dir, raw_data, scenario)):
            for pcd in os.listdir(os.path.join(root_dir, raw_data, scenario, design_id, raw_data_type)): 

                split.append(split_type)

                if raw_data_type == 'PCD' and sub_data_type == 'Image_RGB': 
                    pcd_format = '.ply'
                    img_format = '.png'
                else: 
                    raise ValueError('unknown raw data type')

                if label_type == 'outputJson/Cuboid': 
                    label_format = '.json'
                else: 
                    raise ValueError('unknown label type')

                curr_scenario.append(scenario) 

                pcd_fnames.append(pcd)
                image_fnames.append(pcd.replace(pcd_format, img_format)) 
                pcd_fpaths.append(os.path.join(root_dir, raw_data, scenario, design_id, raw_data_type, pcd))
                image_fpaths.append(os.path.join(root_dir, raw_data, scenario, design_id, sub_data_type, pcd.replace(pcd_format, img_format))) 

                label_fname = pcd.replace(pcd_format, label_format)
                label_fnames.append(label_fname)
                label_fpaths.append(os.path.join(root_dir, label_data, scenario, design_id, label_type, label_fname))    

    df = pd.DataFrame({
        'scenario': curr_scenario, 
        'pcd_fname': pcd_fnames,
        'image_fname': image_fnames, 
        'label_fname': label_fnames, 
        'pcd_path': pcd_fpaths,
        'image_path': image_fpaths, 
        'label_path': label_fpaths,
        'split': split
    }) 

    return df 


def read_ply(filepath): 
    with open(filepath, 'r') as file: 
        while True: 
            line = file.readline().strip() 
            if line == "end_header": 
                break 

        data = [] 
        for line in file: 
            parts = line.split() 
            x, y, z = np.array(parts[:3], dtype=np.float32) * np.array([1, -1, 1])
            intensity = float(parts[6]) 
            data.append([x, y, z, intensity]) 

    return np.array(data, dtype=np.float32) 


def load_cuboid_label(cuboid_path): 

    with open(cuboid_path, 'r') as f: 
        data = f.read() 
        cuboid_data = json.loads(data) 

    names = []
    labels = [] 
    for anno in cuboid_data['annotations']:  

        # 큐보이드의 특정 점들 구하기
        p1 = np.array(list(anno['bBox'][1].values()))
        p0 = np.array(list(anno['bBox'][0].values()))
        p2 = np.array(list(anno['bBox'][2].values())) 
        p4 = np.array(list(anno['bBox'][4].values()))

        height = np.linalg.norm(p0 - p4) / 100 
        width = np.linalg.norm(p0 - p2) / 100
        length = np.linalg.norm(p0 - p1) / 100
        
        center = np.array(list(anno['location'].values())) * np.array([1, -1, 1]) / 100 

        class_name = anno['class']

        rotation = math.radians(anno['rotation']['yaw']) 

        names.append(class_name)
        labels.append(np.concatenate([center, [height, width, length], [rotation]], axis=0).astype(np.float32))

    return names, labels 


def create_data_info_pkl(data_root, data_type, prefix, df, label=True, db=False):
    sep = os.path.sep  # '/'  
    print(f"Processing {data_type} data..") 
    data_infos = df[df['split'] == data_type]

    dss_infos_dict = {} 
    if db: 
        dss_dbinfos_train = {} 
        db_points_saved_path = os.path.join(data_root, f'{prefix}_gt_database') 
        os.makedirs(db_points_saved_path, exist_ok=True) 

    for data_info in tqdm(data_infos.itertuples()): 
        cur_info_dict = {}
        index = data_info.Index 
        pcd_path = data_info.pcd_path
        image_path = data_info.image_path 
        label_path = data_info.label_path 

        # pcd info 
        cur_info_dict['pcd_path'] = pcd_path 

        # image info 
        img = cv2.imread(image_path) 
        image_shape = img.shape[:2] 
        cur_info_dict['image'] = {
            'image_shape': image_shape, 
            'image_path': image_path, 
            'image_idx': int(index), 
        }

        # cuboid info 
        lidar_point = read_ply(pcd_path)
        if label: 
            names, labels = load_cuboid_label(label_path)
            annotation_dict ={}
            annotation_dict['name'] = names 
            annotation_dict['label'] = labels 
            annotation_dict['difficulty'] = None 
            annotation_dict['num_points_in_gt'] = len(lidar_point) 
            cur_info_dict['annos'] = annotation_dict 

            if db: 
                # TODO: create database for data augmentation  
                print('# TODO: create database for data augmentation ')

        dss_infos_dict[int(index)] = cur_info_dict

    saved_path = os.path.join(data_root, f'{prefix}_infos_{data_type}.pkl')
    write_pickle(dss_infos_dict, saved_path) 

    if db: 
        saved_db_path = os.path.join(data_root, f'{prefix}_dbinfos_{data_type}.pkl')
        write_pickle(dss_dbinfos_train, saved_db_path)

    return dss_infos_dict


def main(args, folder_structer):
    data_root = args.data_root
    prefix = args.prefix 

    df = create_dataframe(folder_structer)                                                                          
    df.to_csv(os.path.join(data_root, f'{prefix}_data_info.csv'), index=False)                                                                                                                                                                                                                                                                      

    ## 1. train: create data infomation pkl file && create reduced point clouds 
    ##           && create database(points in gt bbox) for data aumentation
    dss_train_infos_dict = create_data_info_pkl(data_root, 'train', prefix, df, db=False)  # db=True)

    ## 2. val: create data infomation pkl file && create reduced point clouds
    dss_val_infos_dict = create_data_info_pkl(data_root, 'val', prefix, df)
    
    ## 3. trainval: create data infomation pkl file
    dss_trainval_infos_dict = {**dss_train_infos_dict, **dss_val_infos_dict}
    saved_path = os.path.join(data_root, f'{prefix}_infos_trainval.pkl')
    write_pickle(dss_trainval_infos_dict, saved_path)

    ## 4. test: create data infomation pkl file && create reduced point clouds
    dss_test_infos_dict = create_data_info_pkl(data_root, 'test', prefix, df, label=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', default='/workspace/DssDataset', help='your data root for dss')
    parser.add_argument('--prefix', default='dss', help='the prefix name for the saved .pkl file')
    args = parser.parse_args()    
    
    
    folder_structer = {
        'root_dir': args.data_root, 
        'raw_data': 'rawData/Car', 
        'label_data': 'labelingData/Car', 
        'raw_data_type': 'PCD', 
        'sub_data_type': 'Image_RGB', 
        'label_type': 'outputJson/Cuboid', 
        'split': {}  # 자동으로 train : val : test = 8 : 1 : 1 split 
        # 'split': {
        #     'train': ['N01S01M01', 'N02S01M04', 'N04S01M10', 'N09S03M03', 'N11S03M09', 'N16S05M01', 'N18S05M04', 'N19S13M01'], 
        #     'val': ['N23S06M11'], 
        #     'test': ['N25S07M06'],
        # }
    }

    main(args, folder_structer)