import cv2
import numpy as np
import open3d as o3d
import os

o3d.visualization.webrtc_server.enable_webrtc()


def npy2pcd(npy):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy[:, :3])
    density = npy[:, 3]
    colors = [[item, item, item] for item in density]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def pcd2npy(pcd):
    return np.array(pcd.points)


def make_cuboid_from_inference(output_bboxes): 

    cuboids = [] 
    print(type(output_bboxes))
    for bbox in output_bboxes: 
        center, dimension, rotation = bbox[:3], bbox[3:6], bbox[6] 

        height, width, length = dimension

        cuboid = {
            'location': center,
            'dimensions': [height, width, length],
            'rotation_y': rotation
        }

        cuboids.append(cuboid) 

    return cuboids 



def visualize_range_cuboid(point_cloud_range): 
    min_x, min_y, min_z, max_x, max_y, max_z = point_cloud_range 

    vertices = np.array([
        [min_x, min_y, min_z], 
        [max_x, min_y, min_z], 
        [min_x, max_y, min_z], 
        [max_x, max_y, min_z], 
        [min_x, min_y, max_z], 
        [max_x, min_y, max_z], 
        [min_x, max_y, max_z], 
        [max_x, max_y, max_z]
    ])

    edges = [
        [0, 1], 
        [0, 2], 
        [0, 4], 
        [1, 5], 
        [1, 3], 
        [2, 3], 
        [2, 6], 
        [3, 7], 
        [4, 5], 
        [4, 6], 
        [5, 7], 
        [6, 7]
    ]

    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(vertices),
        lines = o3d.utility.Vector2iVector(edges)
    )
    line_set.colors = o3d.utility.Vector3dVector([[1, 1, 1] for _ in range(len(edges))])


    return line_set 


def create_cuboid_lines(center, height, width, length, rotation): 
    '''

           ^ z   x            7 ------ 5
           |   /             / |     / |
           |  /             6 -|---- 4 |   
    y      | /              |  |     | | 
    <------|o               | 3 -----| 1
                            |/   o   |/    
                            2 ------ 0 
    x: front, y: left, z: top
    '''


    half_height = height / 2  # z  
    half_width  = width  / 2  # y 
    half_length = length / 2  # x 

    vertices = np.array([
        [-half_length, -half_width, -half_height],  # 아래 - 오 - 뒤
        [ half_length, -half_width, -half_height],  # 아래 - 오 - 앞 
        [-half_length,  half_width, -half_height],  # 아래 - 왼 - 뒤
        [ half_length,  half_width, -half_height],  # 아래 - 왼 - 앞
        [-half_length, -half_width,  half_height],  #  위 - 오 - 뒤 
        [ half_length, -half_width,  half_height],  #  위 - 오 - 앞 
        [-half_length,  half_width,  half_height],  #  위 - 왼 - 뒤
        [ half_length,  half_width,  half_height]   #  위 - 왼 - 앞 
    ]) 

    edges = [
        [0, 1], 
        [0, 2], 
        [0, 4], 
        [1, 5], 
        [1, 3], 
        [2, 3], 
        [2, 6], 
        [3, 7], 
        [4, 5], 
        [4, 6], 
        [5, 7], 
        [6, 7]
    ]

    # 회전 행렬 적용
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
    vertices = np.dot(vertices, rotation_matrix.T) + center

    return vertices, edges


def visualize_bboxes_3d(bboxes_3d, color=[1, 0, 0]):

    spheres = [] 
    direction_lines = [] 
    line_sets = []  

    for bbox in bboxes_3d: 
        center = bbox['location'] 
        height, width, length = bbox['dimensions']
        rotation = [0, 0, bbox['rotation_y']] 
        vertices, edges = create_cuboid_lines(center, height, width, length, rotation)

        # 큐보이드의 엣지 시각화를 위한 LineSet 생성
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(edges))])
        line_sets.append(line_set)

        
        # 큐보이드의 중심점 시각화를 위한 작은 구 생성
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)  # 구의 반지름을 설정
        sphere.translate(center)  # 구를 큐보이드의 중심으로 이동
        sphere.paint_uniform_color([1, 0, 0])  # 구의 색상을 빨간색으로 설정
        spheres.append(sphere) 

        # 엣지의 끝점에 구 추가
        base_point = vertices[0]
        base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        base_sphere.translate(base_point)
        base_sphere.paint_uniform_color([0, 0, 1])  # 파란색으로 설정
        spheres.append(base_sphere) 

        # 엣지의 끝점에 구 추가
        base_point = vertices[2]
        base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        base_sphere.translate(base_point)
        base_sphere.paint_uniform_color([1, 0, 0])  
        spheres.append(base_sphere)

        # 회전 방향을 나타내는 선 추가
        direction_length = 2.0  # 방향 벡터의 길이
        direction_vector = np.array([direction_length, 0, 0])  # 기본 방향 (x축)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
        rotated_direction = np.dot(rotation_matrix, direction_vector)
        direction_end = center + rotated_direction
        
        # 중심점에서 방향 벡터 끝까지의 선을 생성
        direction_line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([center, direction_end]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        direction_line.paint_uniform_color([0, 1, 0])  # 방향 벡터는 초록색으로 설정
        direction_lines.append(direction_line) 

    return line_sets, spheres, direction_lines 


def vis_core(vis_objs):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    print('vis: ', vis) 
    print('vis.get_view_control(): ', vis.get_view_control()) 

    # point size & background color 설정
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  
    render_option.background_color = np.array([20, 24, 39]) / 255.0

    PAR = os.path.dirname(os.path.abspath(__file__))
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(os.path.join(PAR, 'viewpoint.json'))
    for obj in vis_objs:
        vis.add_geometry(obj)
    ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    vis.destroy_window() 


def visualizer(pc, gt_bboxes=None, output_bboxes=None, point_cloud_range=None):
    '''
    pc: ply or np.ndarray (N, 4)
    gt_bboxes: annotation = {
                    'identifier': identifier, 
                    'name': class_name, 
                    'location': locations, 
                    'dimensions': dimensions, 
                    'rotation_y': rotation_y
                }
    inference_bboxes: (n, )
    '''

    vis_objs = [] 

    # pcd 
    pcd = npy2pcd(pc) if isinstance(pc, np.ndarray) else pc 
    vis_objs.append(pcd)
    
    # global coordinate frame 
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    vis_objs.append(mesh_frame)

    # range cuboid 
    if point_cloud_range is not None: 
        range_line_set = visualize_range_cuboid(point_cloud_range)
        vis_objs.append(range_line_set)

    # 3D bboxes (gt) 
    if gt_bboxes is not None:
        bboxes_line_set, bboxes_spheres, bboxes_direction_line_set = visualize_bboxes_3d(gt_bboxes, color=[0, 1, 0]) 
        vis_objs.extend(bboxes_line_set) 
        vis_objs.extend(bboxes_spheres) 
        vis_objs.extend(bboxes_direction_line_set)
    
    # 3D bboxes (inference)
    if output_bboxes is not None: 
        output_cuboids = make_cuboid_from_inference(output_bboxes)
        bboxes_line_set, bboxes_spheres, bboxes_direction_line_set = visualize_bboxes_3d(output_cuboids, color=[1, 0, 0]) 
        vis_objs.extend(bboxes_line_set) 
        vis_objs.extend(bboxes_spheres) 
        vis_objs.extend(bboxes_direction_line_set)
        
    vis_core(vis_objs)



if __name__ == '__main__':  
    import sys 
    CUR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(CUR))

    from utils import read_ply, read_dss_label, read_dss_label_by_box, inference 

    ply_path = '/workspace/DssDataset/rawData/Car/N01S01M01/Design0018/PCD/000000009325.ply' 
    label_path = '/workspace/DssDataset/labelingData/Car/N01S01M01/Design0018/outputJson/Cuboid/000000009325.json' 
    ckpt_path = '/workspace/PointPillars_Dss/pillar_logs/test/checkpoints/pointpillars_39.pth'
    point_cloud_range=[-40, -40, -3, 40, 40, 1]

    pc = read_ply(ply_path) 

    
    gt_bboxes = read_dss_label_by_box(label_path) if label_path is not None else None 
    output_bboxes, labels, scores = inference(pc, ckpt_path) if ckpt_path is not None else None, None, None 

    visualizer(pc, gt_bboxes=gt_bboxes, output_bboxes=output_bboxes, point_cloud_range=point_cloud_range)