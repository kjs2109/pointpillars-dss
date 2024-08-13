import numpy as np 
import os 
import pickle 
import open3d as o3d 
import json 
import math 


def read_pickle(file_path, suffix='.pkl'): 
    assert os.path.splitext(file_path)[1] == suffix 
    with open(file_path, 'rb') as f: 
        data = pickle.load(f) 
    return data 


def write_pickle(results, file_path): 
    with open(file_path, 'wb') as f: 
        pickle.dump(results, f) 


def read_ply(file_path): 
    with open(file_path, 'r') as file: 
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


def load_pcd_from_ply(ply_path): 
    with open(ply_path, 'r') as f: 

        while True: 
            line = f.readline().strip() 
            if line == "end_header": 
                break 

        points = [] 
        for line in f: 
            parts = line.split() 
            x, y, z = map(float, parts[:3]) 
            points.append([x, -y, z])  

    np_points = np.array(points, dtype=np.float32) 

    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(np_points) 

    return pcd 


def read_dss_label(file_path): 
    with open(file_path, 'r') as f: 
        data = json.load(f) 

    identifier = [] 
    class_name = [] 
    locations = [] 
    dimensions = [] 
    rotation_y = [] 
    for anno in data['annotations']: 
        
        p1 = np.array(list(anno['bBox'][1].values()))
        p0 = np.array(list(anno['bBox'][0].values()))
        p2 = np.array(list(anno['bBox'][2].values())) 
        p4 = np.array(list(anno['bBox'][4].values()))

        height = np.linalg.norm(p0 - p4) / 100 
        width = np.linalg.norm(p0 - p2) / 100
        length = np.linalg.norm(p0 - p1) / 100

        center = np.array(list(anno['location'].values())) * np.array([1, -1, 1]) / 100

        identifier.append(anno['identifier']) 
        class_name.append(anno['class']) 
        locations.append(center) 
        dimensions.append(np.array([height, width, length])) 
        rotation_y.append(math.radians(float(anno['rotation']['yaw'])))
        
    annotation = {
        'identifier': identifier, 
        'name': class_name, 
        'location': locations, 
        'dimensions': dimensions, 
        'rotation_y': rotation_y
    }

    return annotation 

        
def read_dss_label_by_box(file_path): 

    with open(file_path, 'r') as f: 
        cuboid_data = json.load(f) 

    bboxes_3d = [] 
    for anno in cuboid_data['annotations']:  

        p1 = np.array(list(anno['bBox'][1].values()))
        p0 = np.array(list(anno['bBox'][0].values()))
        p2 = np.array(list(anno['bBox'][2].values())) 
        p4 = np.array(list(anno['bBox'][4].values()))

        height = np.linalg.norm(p0 - p4) / 100 
        width = np.linalg.norm(p0 - p2) / 100
        length = np.linalg.norm(p0 - p1) / 100
        
        center = np.array(list(anno['location'].values())) * np.array([1, -1, 1]) / 100 

        bbox_3d = {
            'identifier': anno['identifier'],
            'name': anno['class'],
            'location': center, 
            'dimensions': np.array([height, width, length]), 
            'rotation_y': math.radians(float(anno['rotation']['yaw']))  # degree -> radian 
        }

        bboxes_3d.append(bbox_3d) 

    return bboxes_3d 