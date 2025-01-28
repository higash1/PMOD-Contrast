from h5dataloader.common.structure import *
from h5dataloader.common.common import HDF5DatasetNumpy
from types import MethodType


from typing import Dict, Union, List, Any

import h5py
import os, uuid
import numpy as np

import argparse
import json

# tqdm or rich 
from tqdm import tqdm
# visualization
import cv2

"""Define minibatch key"""

DATASET_CAMERA = 'camera'
DATASET_DEPTH = 'depth'
DATASET_INMAP = 'map'
DATASET_LABEL = 'label'

def parse_src_dst(label_dict:Dict[str, Dict[str, int]], quiet:bool=False) -> List[Dict[str, int]]:
    """parse_src_dst

    ラベル変換用の辞書を生成

    Args:
        label_dict (dict): ラベルの変換情報が記述された辞書
        quiet (bool, optional): Trueの場合, 標準出力を行わない

    Returns:
        list: 相互変換用の辞書のリスト
    """
    dst_list:List[Dict[str, int]] = []
    for key, item in label_dict[CONFIG_TAG_CONVERT].items():
        src_dst:Dict[str, int] = {}
        src_dst[CONFIG_TAG_SRC] = int(key)
        src_dst[CONFIG_TAG_DST] = item
        if quiet is False:
            print('{0:>3d} > {1:>3d}'.format(int(key), item))
        dst_list.append(src_dst)
    return dst_list

def parse_colors(label_dict:Dict[str, Dict[str, Dict[str, List[int]]]]) -> List[Dict[str, Union[str, int, np.ndarray]]]:
    """parse_colors

    ラベルをカラーへ変換する際の辞書を作成

    Args:
        label_dict (dict): ラベルの変換情報が記述された辞書

    Returns:
        List[Dict[str, Union[int, np.ndarray]]]: カラー変換用の辞書のリスト
    """
    dst_list:List[Dict[str, Union[int, np.ndarray]]] = []
    for key, item in label_dict[CONFIG_TAG_DST].items():
        dst_color:Dict[str, Union[str, int, np.ndarray]] = {}
        dst_color[CONFIG_TAG_TAG] = item[CONFIG_TAG_TAG]
        dst_color[CONFIG_TAG_LABEL] = int(key)
        dst_color[CONFIG_TAG_COLOR] = np.array(item[CONFIG_TAG_COLOR], dtype=np.uint8)
        dst_list.append(dst_color)
    return dst_list

def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create minibatch from h5 file')
    parser.add_argument('-hdf5','--hdf5', type=str, metavar='PATH', nargs='+', required=True, help='hdf5 file path')
    parser.add_argument('-c', '--config', type=str, help='config file path')
    
    args: dict = vars(parser.parse_args())

    return args

def find_tf_tree_key(key:str, tf_tree:dict) -> Union[List[str], None]:
    """find_tf_tree_key

    TF-Treeから特定のキー(child_frame_id)を検索し, rootからのキーのリストを生成する

    Args:
        key (str): キー(child_frame_id)
        tf_tree (dict): 検索対象のTF-Tree

    Returns:
        Union[List[str], None]: rootからのキーのリスト. 見つからない場合はNone.
    """
    for tree_key, tree_item in tf_tree.items():
        if key == tree_key:
            return [tree_key]
        key_list = find_tf_tree_key(key, tree_item)
        if isinstance(key_list, list):
            key_list.insert(0, tree_key)
            return key_list

class Createminibatch(HDF5DatasetNumpy):
    def __init__(self, args) -> None:
        # default class variable
        self.block_size:int = 0
        self.quiet:bool = True
        self.visibility_filter_radius:int = 0
            
        self.visibility_filter_threshold:float = 3.0
        self.tf = None
        
        # map points type == 'points'
        self.link_maps:dict = {}
        self.maps:dict = {'map': None}
        
        # config
        self.config = args['config']
        self.output_key = args['hdf5'][0].split('/')[-1].split('.')[0]
        self.output_dir = None
        
        self.label_convert_configs = {}
        self.label_color_configs = {}

    def create_depth_from_pointsmap(self, key:str, link_idx:int, minibatch_config:Dict[str, Union[str, Dict[str, str], List[int], bool, List[float], MethodType]]) -> np.ndarray:
        """create_depth_from_pointsmap

        Pose (Translation & Quaternion) と三次元地図から深度マップを生成する

        Args:
            key (str): HDF5DatasetNumpy.get_key() で生成されたキー
            link_idx (int): linkの番号
            minibatch_config (dict): mini-batchの設定

        Returns:
            np.ndarray: 三次元地図を投影した深度マップ

        Dependent Functions:
            HDF5Dataset.create_pose_from_pose(key, link_idx, minibatch_config)
            HDF5Dataset.depth_common(src, minibatch_config)
        """
        pose:np.ndarray = self.create_pose_from_pose(key, link_idx, minibatch_config)
        dst = self.maps[self.link_maps[str(link_idx)]].create_depthmap(translation=pose[:3], quaternion=pose[-4:], filter_radius=self.visibility_filter_radius, filter_threshold=self.visibility_filter_threshold)

        return self.depth_common(dst, minibatch_config)

    def mkdir(self):
        current_dir:str = '/'.join(__file__.split('/')[:-2])
        self.output_dir:str = os.path.join(current_dir, 'utils','output', self.output_key)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def restoration_src(self, key, src:np.ndarray) -> np.ndarray: 
        return src * self.minibatch[key][CONFIG_TAG_RANGE][1] 
    
    def save_src(self, src:np.ndarray, key:str , index:int, mode='png'):
        output_dir:str = os.path.join(self.output_dir, key)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        save_path:str = os.path.join(output_dir, f'{key}_' + f'{index:06d}.png')
        cv2.imwrite(save_path, src)
        
    def set_hdf5(self, h5_paths:list):
        start_idxs:List[int] = []
        self.h5link_path = os.path.join('/tmp', 'h5dataloader-' + str(uuid.uuid4()) + '-link.hdf5')
        with h5py.File(self.h5link_path, mode='w') as h5link:
            h5_len_tmp:int = 0
            for link_cnt, h5_path in enumerate(h5_paths):
                if os.path.isfile(h5_path) is False:
                    print('Not Found : "%s"', h5_path)
                    exit(1)
                h5link[str(link_cnt)] = h5py.ExternalLink(h5_path, '/') # 番号からlinkを作成
                start_idxs.append(h5_len_tmp)
                with h5py.File(h5_path, mode='r') as h5file:
                    h5_len_tmp += h5file['header/length'][()]
                link_cnt += 1
                self.length = h5_len_tmp
        self.start_idxs = np.array(start_idxs)

        # 一時ファイルを開く
        self.h5links = h5py.File(self.h5link_path, mode='r')
        
    
    def depth2colormap(self, src: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
        out_range = np.where((src < range_min) | (range_max < src))
        src_norm = np.uint8(
            (1.0 - (src - range_min) / (range_max - range_min)) * 255.0)
        colormap = cv2.applyColorMap(src_norm, cv2.COLORMAP_JET)
        colormap[out_range] = [0, 0, 0]
        return colormap

    def get_minibatch(self, index:int):
        link_idx = self.get_link_idx(index=index)
        hdf5_key = self.get_key(index=index)
        
        minibatch_src_dict: Dict[str, np.ndarray] = {}
        for key, minibatch_config in self.minibatch.items():
            # dataType = minibatch_config[CONFIG_TAG_TYPE]
            minibatch:np.ndarray = minibatch_config[CONFIG_TAG_CREATEFUNC](hdf5_key, link_idx, minibatch_config)
            minibatch_src_dict[key] = minibatch
        return minibatch_src_dict
        
    def config_load(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        
        # configファイルをロード -> 辞書型へ
        config_dict:Dict[str, Dict[str, Dict[str, Any]]] = {}
        with open(self.config, mode='r') as configfile:
            config_dict = json.load(configfile)
            
        return config_dict
    

    def convert_label2color(self, src: h5py.Dataset, label_tag:str) -> np.ndarray:
        dst:np.ndarray = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)

        for color_config in self.label_color_configs[label_tag]:
            dst[np.where(src == int(color_config[CONFIG_TAG_LABEL]))] = color_config[CONFIG_TAG_COLOR]
            
        return dst
        
    def visualize(self, src: np.ndarray, demo: bool = False):
        cv2.imshow('src', src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def set_minibach_dict(self, config_dict: Dict[str, Dict[str, Dict[str, Any]]]):
        # tfの設定
        self.tf = config_dict[CONFIG_TAG_TF]

        # データセットの設定
        self.minibatch:Dict[str, Dict[str, Union[str, Dict[str, str], List[int], bool, List[float]]]] = {}     # mini-batchの設定

        for key, item in config_dict[CONFIG_TAG_MINIBATCH].items():
            data_dict:Dict[str, Dict[str, Any]] = {}
            data_dict[CONFIG_TAG_FROM] = item.get(CONFIG_TAG_FROM)
            data_dict[CONFIG_TAG_TYPE] = item.get(CONFIG_TAG_TYPE)

            # ラベルの設定の辞書を生成
            if CONFIG_TAG_LABEL in config_dict:
                for key_label, item_label in config_dict[CONFIG_TAG_LABEL][CONFIG_TAG_CONFIG].items():
                    self.label_convert_configs[key_label] = parse_src_dst(item_label, quiet=True)
                    self.label_color_configs[key_label] = parse_colors(item_label)

            if data_dict[CONFIG_TAG_FROM] is None or data_dict[CONFIG_TAG_TYPE] is None:
                print('keys "from" and "type" must not be null')
                exit(1)

            if isinstance(item.get(CONFIG_TAG_SHAPE), list) is True:
                data_dict[CONFIG_TAG_SHAPE] = tuple(item[CONFIG_TAG_SHAPE])
            else:
                data_dict[CONFIG_TAG_SHAPE] = None

            if isinstance(item.get(CONFIG_TAG_RANGE), list) is True:
                depth_range = []
                depth_range.append(0.0 if item[CONFIG_TAG_RANGE][0] < 0.0 else item[CONFIG_TAG_RANGE][0])
                depth_range.append(np.inf if item[CONFIG_TAG_RANGE][1] is None else item[CONFIG_TAG_RANGE][1])
                data_dict[CONFIG_TAG_RANGE] = tuple(depth_range)
            else:
                data_dict[CONFIG_TAG_RANGE] = DEFAULT_RANGE[data_dict[CONFIG_TAG_TYPE]]

            data_dict[CONFIG_TAG_NORMALIZE] = item.get(CONFIG_TAG_NORMALIZE)
            data_dict[CONFIG_TAG_FRAMEID] = item.get(CONFIG_TAG_FRAMEID)
            data_dict[CONFIG_TAG_LABELTAG] = item.get(CONFIG_TAG_LABELTAG)

            tf_from:str = data_dict[CONFIG_TAG_FROM].get(TYPE_POSE)
            tf_to:str = data_dict[CONFIG_TAG_FRAMEID]
            
            tf_calc:List[Tuple[str, bool]] = []

            if tf_from is not None and tf_to is not None:
                tf_from_list:List[str] = find_tf_tree_key(tf_from, self.tf[CONFIG_TAG_TREE])
                tf_to_list:List[str] = find_tf_tree_key(tf_to, self.tf[CONFIG_TAG_TREE])

                while True:
                    if len(tf_from_list) < 1 or len(tf_to_list) < 1: break
                    if tf_from_list[0] != tf_to_list[0]: break
                    tf_from_list.pop(0)
                    tf_to_list.pop(0)
                tf_from_list.reverse()

                for tf_from_str in tf_from_list:
                    tf_calc.append((tf_from_str, True))
                for tf_to_str in tf_to_list:
                    tf_calc.append((tf_to_str, False))
            data_dict[CONFIG_TAG_TF] = tf_calc

            data_dict[CONFIG_TAG_CREATEFUNC] = self.bind_createFunc(data_dict)

            self.minibatch[key] = data_dict
            
    def create_minibatch(self, h5_paths:List[str]):
        self.set_hdf5(h5_paths)
        self.mkdir()

        config_dict:Dict[str, Dict[str, Dict[str, Any]]] = self.config_load()
        self.set_minibach_dict(config_dict)
        
        for index in tqdm(range(self.length), desc='[ Creating inmap ]', colour='blue', leave=False):
            minibatch_dict = self.get_minibatch(index=index)
            for key, src in minibatch_dict.items():
                if key == DATASET_DEPTH:
                    src:np.ndarray = self.restoration_src(key, src)
                    src = self.depth2colormap(src, self.minibatch[key][CONFIG_TAG_RANGE][0], self.minibatch[key][CONFIG_TAG_RANGE][1])
                elif key == DATASET_INMAP:
                    src:np.ndarray = self.restoration_src(key, src)
                    src = np.where(src > self.minibatch[key][CONFIG_TAG_RANGE][1], 0.0, src)
                    src = (src - self.minibatch[key][CONFIG_TAG_RANGE][0] / (self.minibatch[key][CONFIG_TAG_RANGE][1] - self.minibatch[key][CONFIG_TAG_RANGE][0])) * 255.0
                elif key == DATASET_CAMERA:
                    if self.minibatch[key][CONFIG_TAG_NORMALIZE]:
                        src = src * 255
                    src = src.astype(np.uint8)
                elif key == DATASET_LABEL:
                    src = self.convert_label2color(src, self.minibatch[key][CONFIG_TAG_LABELTAG])
                else:
                    continue
                
                self.save_src(src, key, index, mode='png')
        print(f'Done {self.length} inmap creation. Save to {self.output_key}')
    
if __name__ == '__main__':
    args = arg_parser()
    minibatch = Createminibatch(args)
    minibatch.create_minibatch(args['hdf5'])

