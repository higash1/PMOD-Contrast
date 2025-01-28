import torch.linalg
from pmod.model.utils import quat_inv, quat_mul
from pmod.model.constant import METRIC_SUM_OUTPUT, Q_IDX
from typing import Dict, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class LossNULL(nn.Module):
    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        super(LossNULL, self).__init__()
        self.device = device

    def to(self, device: torch.device) -> None:
        self.device = device

    def forward(self, *args):
        return torch.tensor(0, device=self.device)


class InRange(nn.Module):
    def __init__(self, range_up: float = 1.0, num_classes: int = 0) -> None:
        super(InRange, self).__init__()

        self.range_up: Tensor
        self.register_buffer('range_up', torch.tensor(range_up))

        self.num_classes: Tensor
        self.register_buffer('num_classes', torch.tensor(num_classes))

    def in_range(self, gt: Tensor, seg_gt: Tensor = None) -> Tuple[Tensor, Tensor]:
        in_range = (0. < gt) & (gt < self.range_up)
        if self.num_classes > 0:
            in_range = in_range.squeeze(dim=1)
            in_range_oh: Tensor = F.one_hot(
                seg_gt, num_classes=self.num_classes)
            in_range_oh[~in_range] = False
            return in_range_oh.bool(), torch.sum(in_range_oh, dim=(1, 2))
        else:
            return in_range, torch.sum(in_range, dim=(1, 2, 3))
        
class LossL1(InRange):
    def __init__(self, range_up: float = 1.0) -> None:
        super(LossL1, self).__init__(range_up)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        zeros = torch.zeros_like(gt)
        in_range, cnts = self.in_range(gt)
        l1 = torch.where(in_range, torch.abs(pred - gt), zeros)
        return torch.sum(l1) / torch.sum(cnts)

class ContrastRange(nn.Module):
    def __init__(self, margin: float = 1.0, moldvec:bool = False) -> None:
        super(ContrastRange, self).__init__()
        
        self.margin: Tensor
        self.register_buffer('margin', torch.tensor(margin))
        self.mold_vec: Tensor
        self.register_buffer('mold_vec', torch.tensor(moldvec))
    
    def contrast_range(self, gt_label: Tensor,shape: Tuple[List],label_list:list , pixels_count:bool = False) -> Tensor:
        """
        if mold vector:
        training interpolation in the conv layer,so no need to resize
            gt_label shape (4, 1, 256, 512)
            shape tuple: (4, 1, 256, 512) or easpp raw (4, 256, 256, 512)
            resize is none
        else:
        Shrink gt_label and define triplet range
            gt_label shape (4, 1, 256, 512) -> resize (4, 1, 16, 32)
            shape tuple: (4, 1, 16, 32)
            resize method: nearest neighbor
        """
        init_cond: Tensor = torch.zeros_like(gt_label).bool()
        if self.mold_vec:
            # init_cond: Tensor = torch.logical_or(gt_label == 3, gt_label == 4)
            for label_num in label_list:
                init_cond = torch.logical_or(init_cond, gt_label == label_num)
            dynamic_condition: Tensor = init_cond
            dynamic_condition = dynamic_condition.repeat(1, shape[1], 1, 1)
        else:
            gt_lab = gt_label.float()
            # resize
            resize_gt_label: Tensor = F.interpolate(gt_lab, size=(shape[2], shape[3]), mode='nearest').int()
            for label_num in label_list:
                init_cond: Tensor = torch.logical_or(init_cond, resize_gt_label == label_num)
            
            dynamic_condition: Tensor = init_cond.repeat(1, shape[1], 1, 1)
        
        if pixels_count:
            dynamic_mask: Tensor = dynamic_condition
            dynamic_count: Tensor = torch.sum(torch.sum(init_cond, dim=-1), dim=-1)
            static_mask: Tensor = torch.logical_not(dynamic_mask)
            static_count: Tensor = torch.sum(torch.sum(torch.logical_not(init_cond), dim=-1), dim=-1)
        else:
            label_boolean: Tensor = dynamic_condition.any(dim=-2).any(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            dynamic_mask: Tensor = label_boolean.expand_as(dynamic_condition)
            dynamic_count: Tensor = torch.all(torch.all(init_cond, dim=-1), dim=-1).sum(dim=1, keepdim=True)
            static_mask: Tensor = torch.logical_not(dynamic_mask)
            static_count: Tensor = torch.all(torch.all(torch.logical_not(init_cond), dim=-1), dim=-1).sum(dim=1, keepdim=True)

        return dynamic_mask, static_mask, dynamic_count, static_count
    
class SIlogInRange(nn.Module):
    def __init__(self, range_up: float = 1.0, num_classes: int = 0) -> None:
        super(SIlogInRange, self).__init__()

        self.range_up: Tensor
        self.register_buffer('range_up', torch.tensor(range_up))

        self.num_classes: Tensor
        self.register_buffer('num_classes', torch.tensor(num_classes))

    def in_range(self, gt: Tensor, seg_gt: Tensor = None) -> Tuple[Tensor, Tensor]:
        in_range = (0. < gt) & (gt < self.range_up)
        if self.num_classes > 0:
            in_range = in_range.squeeze(dim=1)
            in_range_oh: Tensor = F.one_hot(
                seg_gt, num_classes=self.num_classes)
            in_range_oh[~in_range] = False
            return in_range_oh.bool(), torch.sum(in_range_oh, dim=(1, 2))
        else:
            return in_range, torch.sum(in_range, dim=(1, 2, 3))
    
class LossSIlog(SIlogInRange):
    def __init__(self, range_up: float = 1.0, variance_focus: float = 0.85) -> None:
        super(LossSIlog, self).__init__(range_up)
        self.variance_focus: float = variance_focus

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        epsilon: float = 1e-6
        in_range, cnts = self.in_range(gt)

        log_diff = torch.log(pred[in_range] + epsilon) - torch.log(gt[in_range] + epsilon)
        
        return torch.sqrt((torch.sum(log_diff ** 2) / torch.sum(cnts)) - (self.variance_focus * ((torch.sum(log_diff) / torch.sum(cnts)) ** 2)))

class LossContrast(ContrastRange):
    def __init__(self, margin: float = 1.0, moldvec:bool = False) -> None:
        super(LossContrast, self).__init__(margin, moldvec)

    def forward(self, pmod_feature: Tensor, inmap_feature: Tensor, gt_label: Tensor, label_list: list, pixels_count:bool = False) -> Tensor:
        """
        pmod feature is the output of the pmod model
            positive
                - pmod feature in static mask
            negative
                - pmod feature in dynamic mask 
        
        static feature is the output of the static inmap model
            anchor
                - inmap feature
            
        reference: PYNet Triplet loss (http://tfsv.tasakilab:5051/fujita/PYNet/blob/292b8fa584bfcf81915862f889ef1e82461d44f1/tools/loss.py#L14)
        """
        
        shape:Tuple = pmod_feature.size()
        
        dynamic_mask, static_mask, dy_cnt, st_cnt = self.contrast_range(gt_label, shape, label_list, pixels_count)
            
        static_loss_l2 = torch.norm((inmap_feature * static_mask) - (pmod_feature * static_mask), p=2, dim=1, keepdim=True)
        dynamic_loss_l2 = torch.norm((inmap_feature * dynamic_mask) - (pmod_feature * dynamic_mask), p=2, dim=1, keepdim=True)
        
        static_loss = torch.sum(static_loss_l2) / torch.sum(st_cnt)
        # if no negative pairs exist -> dynamic loss = 0
        if torch.sum(dy_cnt) == 0:
            dynamic_loss = torch.tensor(0.0)
        else:
            dynamic_loss = torch.sum(dynamic_loss_l2) / torch.sum(dy_cnt)
        return torch.max(static_loss - dynamic_loss + self.margin, torch.zeros_like(static_loss))
    
class QuadroRange(nn.Module):
    def __init__(self, margin: float = 1.0, margin2: float = 0.5, moldvec:bool = False) -> None:
        super(QuadroRange, self).__init__()
        
        self.margin: Tensor
        self.register_buffer('margin', torch.tensor(margin))
        self.margin2: Tensor
        self.register_buffer('margin2', torch.tensor(margin2))
        self.mold_vec: Tensor
        self.register_buffer('mold_vec', torch.tensor(moldvec))
    
    def contrast_range(self, gt_label: Tensor,shape: Tuple[List],label_dict:dict , pixels_count:bool = False) -> Tensor:
        """
        if mold vector:
        training interpolation in the conv layer,so no need to resize
            gt_label shape (4, 1, 256, 512)
            shape tuple: (4, 1, 256, 512) or easpp raw (4, 256, 256, 512)
            resize is none
        else:
        Shrink gt_label and define triplet range
            gt_label shape (4, 1, 256, 512) -> resize (4, 1, 16, 32)
            shape tuple: (4, 1, 16, 32)
            resize method: nearest neighbor
        """
        init_cond: Tensor = torch.zeros_like(gt_label).bool()
        if self.mold_vec:
            # init_cond: Tensor = torch.logical_or(gt_label == 3, gt_label == 4)
            for label_key, label_num in label_dict.items():
                dynamic_cond = torch.logical_or(init_cond, gt_label == label_num)
                if label_key == 'person':
                    person_cond = torch.logical_or(init_cond, gt_label == label_num)
                # elif label_key == 'vehicle':
                    # vehicle_cond = torch.logical_or(init_cond, gt_label == label_num)
                else:
                    ValueError('label_dict key must be person or vehicle')
            person_condition: Tensor = person_cond.repeat(1, shape[1], 1, 1)
            dynamic_condition: Tensor = dynamic_cond.repeat(1, shape[1], 1, 1)
            # vehicle_condition: Tensor = vehicle_cond.repeat(1, shape[1], 1, 1)
        else:
            gt_lab = gt_label.float()
            # resize
            resize_gt_label: Tensor = F.interpolate(gt_lab, size=(shape[2], shape[3]), mode='nearest').int()
            for label_key, label_num in label_dict.items():
                dynamic_cond = torch.logical_or(init_cond, resize_gt_label == label_num)
                if label_key == 'person':
                    person_cond = torch.logical_or(init_cond, resize_gt_label == label_num)
                # elif label_key == 'vehicle':
                    # vehicle_cond = torch.logical_or(init_cond, gt_label == label_num)
                else:
                    ValueError('label_dict key must be person or vehicle')
            dynamic_condition: Tensor = dynamic_cond.repeat(1, shape[1], 1, 1)
            person_condition: Tensor = person_cond.repeat(1, shape[1], 1, 1)
            # vehicle_condition: Tensor = vehicle_cond.repeat(1, shape[1], 1, 1)
        
        if pixels_count:
            person_mask: Tensor = person_condition
            dynamic_mask: Tensor = dynamic_condition
            # vehicle_mask: Tensor = vehicle_condition
            person_count: Tensor = torch.sum(torch.sum(person_cond, dim=-1), dim=-1)
            dynamic_count: Tensor = torch.sum(torch.sum(dynamic_cond, dim=-1), dim=-1)
            # vehicle_count: Tensor = torch.sum(torch.sum(vehicle_cond, dim=-1), dim=-1)
            static_mask: Tensor = torch.logical_not(dynamic_mask)
            static_count: Tensor = torch.sum(torch.sum(torch.logical_not(dynamic_cond), dim=-1), dim=-1)
        else:
            person_label_boolean: Tensor = person_condition.any(dim=-2).any(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            person_mask: Tensor = person_label_boolean.expand_as(person_condition)
            
            dynamic_label_boolean: Tensor = dynamic_condition.any(dim=-2).any(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            dynamic_mask: Tensor = dynamic_label_boolean.expand_as(dynamic_condition)

            # vehicle_label_boolean: Tensor = vehicle_condition.any(dim=-2).any(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            # vehicle_mask: Tensor = vehicle_label_boolean.expand_as(vehicle_condition)

            person_count: Tensor = torch.all(torch.all(person_cond, dim=-1), dim=-1).sum(dim=1, keepdim=True)
            dynamic_count: Tensor = torch.all(torch.all(dynamic_cond, dim=-1), dim=-1).sum(dim=1, keepdim=True)
            # vehicle_count: Tensor = torch.all(torch.all(vehicle_cond, dim=-1), dim=-1).sum(dim=1, keepdim=True)

            static_mask: Tensor = torch.logical_not(dynamic_mask)
            static_count: Tensor = torch.all(torch.all(torch.logical_not(dynamic_cond), dim=-1), dim=-1).sum(dim=1, keepdim=True)

        return person_mask, dynamic_mask, static_mask, person_count, dynamic_count, static_count
    
class LossQuadruplet(QuadroRange):
    def __init__(self, margin: float = 1, margin2:float = 0.5, moldvec: bool = False) -> None:
        super().__init__(margin, margin2, moldvec)
    
    def forward(self, pmod_feature: Tensor, static_feature: Tensor, gt_label: Tensor, label_dict: dict, pixels_count:bool = False) -> Tensor:
        """
        pmod feature is the output of the pmod model
            positive
                - pmod feature in static mask
            negative1
                - pmod feature in dynamic mask
            negative2
                - pmod feature in Person mask -> person is very hard sample 
        
        static feature is the output of the static inmap model
            anchor
                - static feature
            
        reference: PYNet Triplet loss (http://tfsv.tasakilab:5051/fujita/PYNet/blob/292b8fa584bfcf81915862f889ef1e82461d44f1/tools/loss.py#L14)
        quadruplet loss = max(anc pos - anc neg1 + margin, 0) + max(anc neg1 - anc neg2 + margin, 0)
        """
        
        shape:Tuple[int, int, int, int] = pmod_feature.size()
        
        person_mask, dynamic_mask, static_mask, ps_cnt, dy_cnt, st_cnt = self.contrast_range(gt_label, shape, label_dict,pixels_count)
            
        static_loss_l2 = torch.norm((static_feature * static_mask) - (pmod_feature * static_mask), p=2, dim=1, keepdim=True)
        dynamic_loss_l2 = torch.norm((static_feature * dynamic_mask) - (pmod_feature * dynamic_mask), p=2, dim=1, keepdim=True)
        harddynamic_loss_l2 = torch.norm((static_feature * person_mask) - (pmod_feature * person_mask), p=2, dim=1, keepdim=True)
        
        static_loss = torch.sum(static_loss_l2) / torch.sum(st_cnt)
        # if no negative pairs exist -> dynamic loss = 0
        if torch.sum(ps_cnt) == 0:
            harddynamic_loss = torch.tensor(0.0)
        else:
            harddynamic_loss = torch.sum(harddynamic_loss_l2) / torch.sum(ps_cnt)
        if torch.sum(dy_cnt) == 0:
            dynamic_loss = torch.tensor(0.0)
        else:
            dynamic_loss = torch.sum(dynamic_loss_l2) / torch.sum(dy_cnt)
        return torch.max(static_loss - dynamic_loss + self.margin, torch.zeros_like(static_loss)) + torch.max(static_loss - harddynamic_loss + self.margin2, torch.zeros_like(static_loss))
    
class MetricInRange(nn.Module):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricInRange, self).__init__()

        thresholds.sort()
        self.thresholds: Tensor
        self.register_buffer('thresholds', torch.tensor(thresholds))

        self.num_classes: Tensor
        self.register_buffer('num_classes', torch.tensor(num_classes))

    def in_range(self, gt: Tensor, seg_gt: Tensor = None) -> Tuple[Tensor, Tensor]:
        _range_list: List[Tensor] = []
        _prev_th: float = 0.0
        if self.num_classes > 0:
            _one_hot: Tensor = F.one_hot(seg_gt, num_classes=self.num_classes).bool()
        for th_idx in range(self.thresholds.shape[0]):
            _th: Tensor = self.thresholds[th_idx].to(gt.device)
            if _th <= _prev_th:
                continue
            _in_range: Tensor = (_prev_th < gt) & (gt <= _th) & (gt <= 1.0)  # NCHW
            if self.num_classes > 0:
                _in_range: Tensor = torch.squeeze(_in_range, dim=1)  # NHW
                _in_range_oh: Tensor = _one_hot.clone()  # NHWC
                _in_range_oh[~_in_range] = False
                _range_list.append(_in_range_oh.bool().permute(
                    0, 3, 1, 2).unsqueeze(dim=1))  # NTCHW
            else:
                _range_list.append(_in_range.unsqueeze(dim=1))  # NTCHW
            _prev_th = _th.clone()

        range_cat: Tensor = torch.cat(_range_list, dim=1)  # NTCHW
        range_cnt: Tensor = range_cat.sum(
            dim=(3, 4)) if self.num_classes > 0 else range_cat.sum(dim=(2, 3, 4))
        return range_cat, range_cnt


class MetricBatchSumAPE(MetricInRange):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricBatchSumAPE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        in_range, cnts = self.in_range(gt, seg_gt)
        ape = torch.abs(pred - gt) / gt  # NCHW

        ape = ape.unsqueeze(dim=1).expand_as(in_range)  # NTCHW
        zeros = torch.zeros_like(ape)
        ape = torch.where(in_range, ape, zeros)

        if self.num_classes > 0:
            return METRIC_SUM_OUTPUT(ape.sum(dim=(3, 4)), cnts)  # NTC
        else:
            return METRIC_SUM_OUTPUT(ape.sum(dim=(2, 3, 4)), cnts)  # NT


class MetricSumAPE(MetricBatchSumAPE):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricSumAPE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        ape = super().forward(pred, gt, seg_gt)
        return METRIC_SUM_OUTPUT(torch.sum(ape.sum, dim=0), torch.sum(ape.count, dim=0))


class MetricIntersection(nn.Module):
    def __init__(self):
        super(MetricIntersection, self).__init__()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        num_classes = pred.shape[1]

        pred = torch.argmax(pred, dim=1)
        pred = F.one_hot(pred, num_classes=num_classes)
        gt = F.one_hot(gt, num_classes=num_classes)

        intersection = torch.logical_and(pred, gt)

        return torch.sum(intersection, dim=(1, 2))


class MetricSumIntersection(MetricIntersection):
    def __init__(self):
        super(MetricSumIntersection, self).__init__()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        return torch.sum(super().forward(pred, gt), dim=0)


class MetricUnion(nn.Module):
    def __init__(self, smooth: float = 1e-10):
        super(MetricUnion, self).__init__()
        self.smooth = torch.tensor(smooth)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        num_classes = pred.shape[1]

        pred = torch.argmax(pred, dim=1)
        pred = F.one_hot(pred, num_classes=num_classes)
        gt = F.one_hot(gt, num_classes=num_classes)

        union = torch.logical_or(pred, gt)

        return torch.sum(union, (1, 2)) + self.smooth


class MetricSumUnion(MetricUnion):
    def __init__(self, smooth: float = 1e-10):
        super().__init__(smooth)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        return torch.sum(super().forward(pred, gt) - self.smooth, dim=0) + self.smooth


class MetricBatchSumAE(MetricInRange):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricBatchSumAE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        in_range, cnts = self.in_range(gt, seg_gt)
        ae = torch.abs(pred - gt)  # NCHW

        ae = ae.unsqueeze(dim=1).expand_as(in_range)  # NTCHW
        zeros = torch.zeros_like(ae)
        ae = torch.where(in_range, ae, zeros)

        if self.num_classes > 0:
            return METRIC_SUM_OUTPUT(ae.sum(dim=(3, 4)), cnts)  # NTC
        else:
            return METRIC_SUM_OUTPUT(ae.sum(dim=(2, 3, 4)), cnts)  # NT


class MetricSumAE(MetricBatchSumAE):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricSumAE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        ae = super().forward(pred, gt, seg_gt)
        return METRIC_SUM_OUTPUT(torch.sum(ae.sum, dim=0), torch.sum(ae.count, dim=0))


class MetricBatchSumSE(MetricInRange):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricBatchSumSE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        in_range, cnts = self.in_range(gt, seg_gt)
        se = torch.pow(pred - gt, 2)  # NCHW

        se = se.unsqueeze(dim=1).expand_as(in_range)  # NTCHW
        zeros = torch.zeros_like(se)
        se = torch.where(in_range, se, zeros)

        if self.num_classes > 0:
            return METRIC_SUM_OUTPUT(se.sum(dim=(3, 4)), cnts)  # NTC
        else:
            return METRIC_SUM_OUTPUT(se.sum(dim=(2, 3, 4)), cnts)  # NT


class MetricSumSE(MetricBatchSumSE):
    def __init__(self, thresholds: List[float] = [1.0], num_classes: int = 0) -> None:
        super(MetricSumSE, self).__init__(thresholds, num_classes)

    def forward(self, pred: Tensor, gt: Tensor, seg_gt: Tensor = None) -> METRIC_SUM_OUTPUT:
        se = super().forward(pred, gt, seg_gt)
        return METRIC_SUM_OUTPUT(torch.sum(se.sum, dim=0), torch.sum(se.count, dim=0))