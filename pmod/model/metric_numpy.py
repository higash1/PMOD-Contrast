"""
result2reevaluate用のnumpyを使ったメトリクス計算

Using:
    data.hdf5
        GT-Label, GT-Depth : 0~num_classes-1, 0~depth_norm_range_max (0~20, 0~80)
        Pred-Label, Pred-Depth : 0~num_classes-1, 0~depth_norm_range_max (0~20, 0~80)   

mIoU [%], MAE [m], RMSE [m], MAPE ~= (1m あたりの距離誤差)
"""

import numpy as np
from openpyxl.worksheet.worksheet import Worksheet
import openpyxl as xl

from typing import List, Tuple, Dict, Union, Any, NamedTuple

class METRIC_SUM_OUTPUT(NamedTuple):
    sum: np.ndarray
    count: np.ndarray

class METRIC_RESULT_NUMPY:
    def __init__(self, name: str = '', use_class: bool = False, use_thresholds: bool = False, all_tag: str = 'All') -> None:
        self.name: str = name
        self.use_class: bool = use_class
        self.use_thresholds: bool = use_thresholds
        self.all_tag: str = all_tag

        self.all_metric: np.ndarray = None
        self.class_metric: np.ndarray = None
        self.threshold_metric: np.ndarray = None

        self.counts_metric: np.ndarray = None
        self.sum_class_metric: np.ndarray = None
        self.sum_threshold_metric: np.ndarray = None

    def add(self, count_metric: np.ndarray = None, class_metric: np.ndarray = None, threshold_metric: np.ndarray = None):
        if isinstance(count_metric, np.ndarray):
            if self.counts_metric is None:
                self.counts_metric = np.zeros_like(count_metric)
            self.counts_metric = self.counts_metric + count_metric

        if isinstance(class_metric, np.ndarray):
            tmp_class_metric = class_metric
        elif isinstance(self.class_metric, np.ndarray):
            tmp_class_metric = self.class_metric
        else:
            tmp_class_metric = None
        if tmp_class_metric is not None:
            if self.sum_class_metric is None:
                self.sum_class_metric = np.zeros_like(tmp_class_metric)
            self.sum_class_metric = self.sum_class_metric + tmp_class_metric

        if isinstance(threshold_metric, np.ndarray):
            tmp_threshold_metric = threshold_metric
        elif isinstance(self.threshold_metric, np.ndarray):
            tmp_threshold_metric = self.threshold_metric
        else:
            tmp_threshold_metric = None
        if tmp_threshold_metric is not None:
            if self.sum_threshold_metric is None:
                self.sum_threshold_metric = np.zeros_like(
                    tmp_threshold_metric)
            self.sum_threshold_metric = self.sum_threshold_metric + tmp_threshold_metric

class MetricIntersection:
    def __init__(self):
        super(MetricIntersection, self).__init__()
    
    def __call__(self, pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
        pred = np.eye(num_classes)[pred]
        gt = np.eye(num_classes)[gt]

        intersection = np.logical_and(pred, gt) # HWC

        return np.sum(intersection, axis=(0, 1))
        
class MetricUnion:
    def __init__(self, smooth: float = 1e-10):
        super(MetricUnion, self).__init__()
        self.smooth = smooth
    
    def __call__(self, pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
        pred = np.eye(num_classes)[pred]
        gt = np.eye(num_classes)[gt]

        union = np.logical_or(pred, gt)

        return np.sum(union, axis=(0, 1)) + self.smooth
    
class MetricInRange:
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricInRange, self).__init__()

        self.thresholds = np.array(sorted(thresholds))
        self.num_classes = num_classes

    def in_range(self, gt: np.ndarray, seg_gt: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        range_list: List[np.ndarray] = []
        prev_th = 0.0
        if self.num_classes > 0:
            one_hot = np.eye(self.num_classes)[seg_gt]
        for th_idx in range(self.thresholds.shape[0]):
            th = self.thresholds[th_idx]
            if th <= prev_th:
                continue
            in_range = (prev_th < gt) & (gt <= th) & (gt <= 1.0)
            if self.num_classes > 0:
                in_range_oh:np.ndarray = one_hot.copy()
                in_range_oh[~in_range] = False # HWC
                in_range_oh_bool:np.ndarray = (np.expand_dims(in_range_oh.astype(np.bool8), axis=0)) # THWC
                range_list.append(in_range_oh_bool.transpose([0, 3, 1, 2])) # TCHW
            else:
                raise NotImplementedError
                range_list.append(in_range.reshape(-1, 1, *in_range.shape[1:]))
            prev_th = th
        
        range_cat = np.concatenate(range_list, axis=0) # TCHW
        range_cnt = np.sum(range_cat, axis=(2, 3)) # TC
        return range_cat, range_cnt
    
class MetricBatchSumAE(MetricInRange):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricBatchSumAE, self).__init__(thresholds, num_classes)

    def __call__(self, pred: np.ndarray, gt: np.ndarray, seg_gt: np.ndarray = None) -> METRIC_SUM_OUTPUT:
        in_range, cnts = self.in_range(gt, seg_gt)
        ae: np.ndarray = np.expand_dims(np.abs(pred - gt), axis=0)  # CHW

        ae = np.broadcast_to(np.expand_dims(ae, axis=0), in_range.shape) # TCHW
        zeros = np.zeros_like(ae)
        ae = np.where(in_range, ae, zeros)

        return METRIC_SUM_OUTPUT(ae.sum(axis=(2, 3)), cnts)  # TC
    
class MetricBatchSumSE(MetricInRange):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super().__init__(thresholds, num_classes)
    
    def __call__(self, pred: np.ndarray, gt: np.ndarray, seg_gt: np.ndarray = None) -> METRIC_SUM_OUTPUT:
        in_range, cnts = self.in_range(gt, seg_gt)
        se = np.expand_dims(np.power(pred - gt, 2), axis=0) # CHW
        
        se = np.broadcast_to(np.expand_dims(se, axis=0), in_range.shape) # TCHW
        zeros = np.zeros_like(se)
        se = np.where(in_range, se, zeros)
        
        return METRIC_SUM_OUTPUT(se.sum(axis=(2, 3)), cnts)  # TC

class MetricBatchSumAPE(MetricInRange):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricBatchSumAPE, self).__init__(thresholds, num_classes)

    def __call__(self, pred: np.ndarray, gt: np.ndarray, seg_gt: np.ndarray = None) -> METRIC_SUM_OUTPUT:
        in_range, cnts = self.in_range(gt, seg_gt)
        ape = np.expand_dims(np.abs(pred - gt) / gt, axis=0)  # CHW

        ape = np.broadcast_to(np.expand_dims(ape, axis=0), in_range.shape) # TCHW
        zeros = np.zeros_like(ape)
        ape = np.where(in_range, ape, zeros)

        return METRIC_SUM_OUTPUT(ape.sum(axis=(2, 3)), cnts) # TC
