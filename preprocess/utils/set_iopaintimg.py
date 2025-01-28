import h5py
import numpy as np

from typing import Dict, Union, List, Any

from h5datacreator import *
import shutil, re
from tqdm import tqdm
import argparse

from output_minibatch import Createminibatch

import cv2


def argment_parser() -> dict:
    parser = argparse.ArgumentParser(description="KITTI360 evaluation")
    parser_setiopaint_img = parser.add_argument_group('SET_IOPAINTIMG')
    parser_setiopaint_img.add_argument('-hdf5','--hdf5', type=str, metavar='PATH', nargs='+', required=True, help='PATH of training HDF5 datasets.')
    parser_setiopaint_img.add_argument('-d', '--dir', type=str, required=True, help='directory of iopaintimg')
    
    parser_create_minibatch = parser.add_argument_group('CREATE_MINIBATCH')
    parser_create_minibatch.add_argument('-c', '--config', type=str, help='config file path')
    args: dict = vars(parser.parse_args())
    
    return args

class SET_IOPAINTIMG(Createminibatch):
    def __init__(self, args):
        super(SET_IOPAINTIMG, self).__init__(args)
        self.args = args
        
        self.set_hdf5(self.args['hdf5'])
        config_dict:Dict[str, Dict[str, Dict[str, Any]]] = self.config_load()
        self.set_minibach_dict(config_dict)
        
        origin_path: str = self.args['dir']
        iopaint_list:List[str] = self.get_iopaintimg_list(origin_path)
        
        with h5py.File(self.args['hdf5'][0], 'r') as f:
            self.header = int(f['header/length'][()])

        self.add_hdf5_read()
        for itr, ioimg_path in tqdm(enumerate(iopaint_list), desc="set iopaint_img", colour='blue', leave=False):
            iopaint_img:np.ndarray = self.get_iopaint_img(os.path.join(origin_path, ioimg_path))  
            
            self.add_hdf5(itr, iopaint_img)

        print(f"complete {self.args['hdf5'][0]} set iopaint_img!!")
        
    def add_hdf5_read(self):
        save_path = f"{self.args['hdf5'][0].split('.')[0]}_iopaint.hdf5"

        if os.path.exists(save_path):
            os.remove(save_path)
            shutil.copyfile(self.args['hdf5'][0], save_path)
        else:
            shutil.copyfile(self.args['hdf5'][0], save_path)
        self.iopaint_img_hdf5 = h5py.File(save_path, 'a')
    
    def get_iopaintimg_list(self, sequence_path: str) -> List[str]:
        return sorted(os.listdir(sequence_path))
    
    def get_iopaint_img(self, iopaint_imgpath:str) -> np.ndarray:
        iopaint_img:np.ndarray = cv2.imread(iopaint_imgpath, cv2.IMREAD_ANYCOLOR)
        return iopaint_img
    
    def add_hdf5(self, itr:int, iopaint_img:np.ndarray):
        set_bgr8(self.iopaint_img_hdf5[f'data/{itr}'], 'iopaintimg', iopaint_img, frame_id='cam0')
        
    
        
    
if __name__ == '__main__':
    args = argment_parser()
    SET_IOPAINTIMG(args)