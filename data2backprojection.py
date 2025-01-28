import numpy as np
import open3d as o3d

import os
import h5py
import json, cv2
import argparse

from tqdm import tqdm
from pointsmap import Points

PMOD_HEIHGT:int = 256
PMOD_WIDTH:int = 512

def argment_parser():
    parser = argparse.ArgumentParser(description='Least Square Scale')
    parser.add_argument('--hdf5', '-hdf5', type=str, required=True, help='input hdf5 file')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='dataset name carla or kitti or real')
    parser.add_argument('--config', '-c', type=str, required=True, help='config file path')
    parser.add_argument('--name', '-n', type=str, required=True, help='output name')
    
    parser.add_argument('--remove', '-rm', action='store_true', help='remove corresponding point cloud by label')
    parser.add_argument('--pointcloud', '-pc', action='store_true', help='not create point cloud')
    return parser.parse_args()


class nddepth_pointcloud:
    def __init__(self,args):
        self.args = args
        
        # hdf5 file
        self.h5links = h5py.File(self.args.hdf5, 'r')
        
        # output name
        self.sequence_name:str = self.args.name

        # dataset name
        self.dataset_name:str = self.args.dataset

        # original image size
        self.img_height:int = PMOD_HEIHGT
        self.img_width:int = PMOD_WIDTH
        
        # define each img
        self.gt_depth: np.ndarray = None
        self.pred_depth: np.ndarray = None
        self.gt_seg: np.ndarray = None
        self.pred_seg: np.ndarray = None
        self.img: np.ndarray = None
        
        # intrinsic matrix
        self.cam_K: np.ndarray = np.eye(3)
        self.inv_cam_K: np.ndarray = None
        
        header_length:int = int(self.h5links['header/length'][()])
        
        # save dir
        self.save_dir = os.path.join(os.path.dirname(self.args.hdf5),self.sequence_name)
        self.quarity_dir = os.path.join(self.save_dir,'qualitative')

        if not os.path.exists(self.quarity_dir):
            os.makedirs(self.quarity_dir)

        os.makedirs(os.path.join(self.quarity_dir,'pred_depth'),exist_ok=True)
        os.makedirs(os.path.join(self.quarity_dir,'pred_seg'),exist_ok=True)
        os.makedirs(os.path.join(self.quarity_dir,'gt_depth'),exist_ok=True)
        os.makedirs(os.path.join(self.quarity_dir,'gt_seg'),exist_ok=True)
        # os.makedirs(os.path.join(self.quarity_dir,'inmap'),exist_ok=True)
        os.makedirs(os.path.join(self.quarity_dir,'img'),exist_ok=True)
        
        if self.args.pointcloud:
            self.pc_dir = os.path.join(self.save_dir,'point_cloud')
            if not os.path.exists(self.pc_dir):
                os.makedirs(self.pc_dir)
            os.makedirs(os.path.join(self.pc_dir, 'gt'), exist_ok=True)
            os.makedirs(os.path.join(self.pc_dir, 'pred'), exist_ok=True)
            os.makedirs(os.path.join(self.pc_dir, 'inmap'), exist_ok=True)
            
        
        # json loader
        intrinsic_key = self.json_loader()
        # intrinsic loader
        self.intrinsic_loader(intrinsic_key)
        
        
        if self.dataset_name == 'real':
            max_depth:float = 20.0
        else:
            max_depth:float = 100.0
            
        gt_pts = Points(quiet=True)
        gt_pts.set_intrinsic(self.cam_K)
        gt_pts.set_shape((self.img_height, self.img_width))
        gt_pts.set_depth_range((0, max_depth))
        
        pred_pts = Points(quiet=True)
        pred_pts.set_intrinsic(self.cam_K)
        pred_pts.set_shape((self.img_height, self.img_width))
        pred_pts.set_depth_range((0, max_depth))
        
        inmap_pts = Points(quiet=True)
        inmap_pts.set_intrinsic(self.cam_K)
        inmap_pts.set_shape((self.img_height, self.img_width))
        inmap_pts.set_depth_range((0, max_depth))

        for idx in tqdm(range(header_length), desc='[ Creating scale txt ]', colour='blue'):
            self.hdf5_loader(idx)
            # save model prediction
            self.save_modelprediction(idx, key='gt')
            self.save_modelprediction(idx, key='pred')
            self.save_img(idx)
            
            
            if not self.args.pointcloud:
                continue

            gt_pts.set_depthmap_semantic2d(self.gt_depth, self.gt_seg)
            pred_pts.set_depthmap_semantic2d(self.pred_depth, self.gt_seg)
            inmap_pts.set_depthmap_semantic2d(self.in_map, self.gt_seg)
            
            gt_points, gt_semantic1d = gt_pts.get_semanticpoints()
            pred_points, pred_semantic1d = pred_pts.get_semanticpoints()
            inmap_points, inmap_semantic1d = inmap_pts.get_semanticpoints()

            if self.args.remove:
                gt_points, gt_semantic1d = self.remove_pointcloud(gt_points, gt_semantic1d)
                pred_points, pred_semantic1d = self.remove_pointcloud(pred_points, pred_semantic1d)
                inmap_points, inmap_semantic1d = self.remove_pointcloud(inmap_points, inmap_semantic1d)

            gt_semantic3d:np.ndarray = self.create_semantic3d_from_semantic1d(gt_semantic1d)
            pred_semantic3d:np.ndarray = self.create_semantic3d_from_semantic1d(pred_semantic1d)
            inmap_semantic3d:np.ndarray = self.create_semantic3d_from_semantic1d(inmap_semantic1d)

            # save point cloud
            self.save_pointcloud(gt_points, gt_semantic3d, idx, key='gt')
            self.save_pointcloud(pred_points, pred_semantic3d, idx, key='pred')
            self.save_pointcloud(inmap_points, inmap_semantic3d, idx, key='inmap')
        
    def intrinsic_loader(self, intrinsic_key: str = None):
        if self.args.dataset == 'kitti':
            temp_h5links = h5py.File('/workspace/pmod/datasets/kitti/train_500/kitti360_seq00_train.hdf5', 'r')
        elif self.args.dataset == 'carla':
            temp_h5links = h5py.File('/workspace/pmod/datasets/carla/test/Town01_ClearNoon_500_5000_test.h5', 'r')
        elif self.args.dataset == 'real':
            temp_h5links = h5py.File('/workspace/pmod/datasets/real/train/cross1.hdf5', 'r')
        else:
            raise ValueError('Invalid dataset name')
        
        if intrinsic_key is None:
            raise ValueError('Invalid intrinsic key')
            
        Cx:float = temp_h5links[intrinsic_key]['Cx'][()]
        Cy:float = temp_h5links[intrinsic_key]['Cy'][()]
        Fx:float = temp_h5links[intrinsic_key]['Fx'][()]
        Fy:float = temp_h5links[intrinsic_key]['Fy'][()]
        height:float = temp_h5links[intrinsic_key]['height'][()]
        width:float = temp_h5links[intrinsic_key]['width'][()]
        
        height_threshold = PMOD_HEIHGT / height
        width_threshold = PMOD_WIDTH / width

        self.cam_K[0,0] = Fx * width_threshold
        self.cam_K[1,1] = Fy * height_threshold
        self.cam_K[0,2] = Cx * width_threshold
        self.cam_K[1,2] = Cy * height_threshold
        
        self.inv_cam_K = np.linalg.inv(self.cam_K)
            
    def json_loader(self):
        # json list setting
        config_json_open = open(self.args.config,'r')
        config_json_load = json.load(config_json_open)
        if self.args.dataset == 'kitti' or self.args.dataset == 'nyu':
            self.dst_list:list = list(config_json_load['label']['config']['5class']['dst'].items())
        elif self.args.dataset == 'carla':
            self.dst_list:list = list(config_json_load['label']['config']['carla-0.9.14']['dst'].items())
        elif self.args.dataset == 'real':
            self.dst_list:list = list(config_json_load['label']['config']['label']['dst'].items())
            
        intrinsic_key: str = config_json_load['mini-batch']['map']['from']['intrinsic']
        
        return intrinsic_key
                    
    def hdf5_loader(self, idx):
        """
        hdf5 loader 
        
        result data.hdf5 -> pred_depth, pred_seg, gt_depth, gt_seg
        key name -> Pred-Depth, Pred-Label, GT-Depth, GT-Label
        """
        self.gt_depth = self.h5links[f'data/{idx}/GT-Depth'][()]
        self.pred_depth = self.h5links[f'data/{idx}/Pred-Depth'][()]
        self.gt_seg = self.h5links[f'data/{idx}/GT-Label'][()]
        self.in_map = self.h5links[f'data/{idx}/Input-Map'][()]
        self.pred_seg = self.h5links[f'data/{idx}/Pred-Label'][()]
        self.img = self.h5links[f'data/{idx}/Input-Camera'][()]

    
    def remove_pointcloud(self, points:np.ndarray, semantic1d:np.ndarray) -> np.ndarray:
        if self.dataset_name == 'carla':
            rm_label_list:list = [0, 1, 2, 3, 9, 12, 13, 14]
        elif self.dataset_name == 'kitti' or self.dataset_name == 'real':
            rm_label_list:list = [0, 2]
            
        rm_label_condition = np.zeros_like(semantic1d, dtype=bool)  # 初期化
        for label in rm_label_list:
            add_condition = (semantic1d == label)
            rm_label_condition = np.logical_or(rm_label_condition, add_condition)
        
        points = points[np.logical_not(rm_label_condition)]
        semantic1d = semantic1d[np.logical_not(rm_label_condition)]
        return points, semantic1d

    def create_semantic3d_from_semantic1d(self, semantic1d: np.ndarray) -> np.ndarray:
        """ semantic 3d ndarray from semantic 1d ndarray 
        Args:
            semantic1d (np.ndarray): semantic 1d ndarray
        reference:
            self.dst_list: list of tuple ('0', {'tag': 'road', 'color': [128, 64, 128]})
        """ 
        semantic1d_color = np.zeros((semantic1d.shape[0], 3),dtype=np.float32)
        
        for color_config in self.dst_list:
            semantic1d_color[np.where(semantic1d == int(color_config[0]))] = np.array(color_config[1]['color'][::-1]) / 255.0
            
        return semantic1d_color
            
    def depth2colormap(self, range_min: float, range_max: float, key:str = 'gt') -> np.ndarray:
        if key == 'gt':
            src = self.gt_depth
        elif key == 'pred':
            src = self.pred_depth
            
        out_range = np.where((src < range_min) | (range_max < src))
        src_norm = np.uint8(
            (1.0 - (src - range_min) / (range_max - range_min)) * 255.0)
        colormap = cv2.applyColorMap(src_norm, cv2.COLORMAP_JET)
        colormap[out_range] = [0, 0, 0]
        return colormap

    def convert_label(self, key:str = 'gt') -> np.ndarray:
        if key == 'gt':
            src = self.gt_seg
        elif key == 'pred':
            src = self.pred_seg
        
        color_label: np.ndarray = np.zeros(
            (src.shape[0], src.shape[1], 3), dtype=np.uint8)

        for color_config in self.dst_list:
            color_label[np.where(src == int(color_config[0])) 
                        ] = color_config[1]['color']
        return color_label
            
    def save_pred_depth(self, pred_depth, frame_num):
        pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min()) * 255
        cv2.imwrite(os.path.join(self.pred_depthdir, f'{frame_num:06d}_pred.png'), pred_depth, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
    def save_modelprediction(self, frame_num:int, key:str = 'gt'):
        depth:np.ndarray = self.depth2colormap(0, 100, key)
        colmap:np.ndarray = self.convert_label(key)
        cv2.imwrite(os.path.join(self.quarity_dir, f'{key}_depth', f'{key}_{frame_num:06d}.png'), depth, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(os.path.join(self.quarity_dir, f'{key}_seg', f'{key}_{frame_num:06d}.png'), colmap, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    def save_img(self, frame_num:int):
        cv2.imwrite(os.path.join(self.quarity_dir, 'img', f'img_{frame_num:06d}.png'), self.img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    def save_pointcloud(self, points:np.ndarray, colors:np.ndarray, frame_num:int, key:str = 'gt'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(self.pc_dir, key, f'{key}_{frame_num:06d}.ply'), pcd, write_ascii=True)
        
if __name__ == "__main__":
    args = argment_parser()
    nddepth_pointcloud(args)
    print('Done')