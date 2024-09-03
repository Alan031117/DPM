import colorlog as logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import math
import torch
import open3d as o3d
from torch import Tensor
from typing import Dict, List, Tuple
from system.modules.utils import PoseTool


class Recorder:
    def __init__(self):
        self.record_dict: Dict[str, List] = {}
        self.reduction_func = {
            'min': self.min,
            'max': self.max,
            'mean': self.mean,
            'best': self.best,
            'none': lambda x: x
        }

    def add_dict(self, metric_dict: dict):
        for key, value in metric_dict.items():
            if key not in self.record_dict.keys():
                self.record_dict[key] = []
            self.record_dict.get(key).append(value)

    def add_item(self, key: str, value):
        if key not in self.record_dict.keys():
            self.record_dict[key] = []
        self.record_dict.get(key).append(value)

    def mean(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                return_dict[key] = sum(value) / len(value)
        return return_dict

    def max(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                return_dict[key] = max(value)
        return return_dict

    def min(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                return_dict[key] = min(value)
        return return_dict

    def best(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                if value[0] > value[-1]:
                    return_dict[key] = min(value)
                else:
                    return_dict[key] = max(value)
        return return_dict

    def tostring(self, reduction='best') -> str:
        assert reduction in ['min', 'max', 'mean', 'best', 'none']
        reduction_dic = self.reduction_func.get(reduction)()
        string = ''
        if len(reduction_dic) > 0:
            for key, value in reduction_dic.items():
                if isinstance(value, list):
                    value_str = value
                else:
                    value_str = f'{value:4.5f}'
                string += f'\t{key:<20s}: ({value_str})\n'
            string = '\n' + string
        return string

    def clear(self):
        self.record_dict.clear()


class Optimizer:
    def __init__(self, args):
        self.name = args.type.lower()
        self.kwargs = args.kwargs
        if self.name == 'adamw':
            self.optimizer = torch.optim.AdamW
        elif self.name == 'adam':
            self.optimizer = torch.optim.Adam
        elif self.name == 'sgd':
            self.optimizer = torch.optim.SGD
        else:
            raise NotImplementedError

    def __call__(self, parameters):
        return self.optimizer(params=parameters, **self.kwargs)


class IdentityScheduler(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def step(self):
        pass


class Scheduler:
    def __init__(self, args):
        self.name = args.type.lower()
        self.kwargs = args.kwargs
        if self.name == 'identity':
            self.scheduler = IdentityScheduler
        elif self.name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        elif self.name == 'cosine_restart':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        else:
            raise NotImplementedError

    def __call__(self, optimizer):
        return self.scheduler(optimizer=optimizer, **self.kwargs)


class fakecast:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def try_load_state_dict(model, state_dict, name='model', log=True):
    model_keys = model.state_dict().keys()
    file_keys = state_dict.keys()
    if model_keys == file_keys:
        try:
            model.load_state_dict(state_dict)
            if log:
                logger.info(f"{name} loaded successfully.")
            return
        except:
            if log:
                logger.warning(f"{name} loaded failed.")
            return
    else:
        missing = model_keys - file_keys
        warnings_str = f'{name} loaded with {len(model_keys)} in model, {len(file_keys)} in file.\n'
        if len(missing) != 0:
            warnings_str += f"{len(missing)} missing parameters (in model):\n" + ", ".join(missing) + '\n'
        unexpected = file_keys - model_keys
        if len(unexpected) != 0:
            warnings_str += f"{len(unexpected)} unexpected parameters (in file):\n" + ", ".join(
                unexpected) + '\n'
        try:
            model.load_state_dict(state_dict, strict=False)
            if log:
                logger.warning(warnings_str)
            return
        except:
            if log:
                logger.warning(f"{name} loaded failed.")
            return


def icp_refinement(src: Tensor, dst: Tensor, init_R: Tensor, init_T: Tensor) -> Tuple[Tensor, Tensor]:
    """
    icp修正刚体变换矩阵

    :param src: (B, 3, N)
    :param dst: (B, 3, N)
    :param init_R: (B, 3, 3)
    :param init_T: (B, 3, 1)
    :return: (B, 3, 3), (B, 3, 1)
    """
    B, device = init_R.shape[0], init_R.device
    src_pcd_o3d = o3d.geometry.PointCloud()
    dst_pcd_o3d = o3d.geometry.PointCloud()

    src_list = src.detach().cpu().numpy()
    dst_list = dst.detach().cpu().numpy()
    init_SE3_list = np.repeat(np.eye(4)[np.newaxis, :, :], axis=0, repeats=B)
    init_SE3_list[:, :3, :3] = init_R.detach().cpu().numpy()
    init_SE3_list[:, :3, 3:] = init_T.detach().cpu().numpy()

    SE3_refinement = []
    for src_pcd, dst_pcd, init_SE3 in zip(src_list, dst_list, init_SE3_list):
        src_pcd_o3d.points = o3d.utility.Vector3dVector(src_pcd.T)
        dst_pcd_o3d.points = o3d.utility.Vector3dVector(dst_pcd.T)
        icp = o3d.pipelines.registration.registration_icp(
            source=src_pcd_o3d, target=dst_pcd_o3d, max_correspondence_distance=1.0, init=init_SE3,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
        icp_SE3 = icp.transformation
        delta_pose = np.linalg.inv(icp_SE3) @ init_SE3
        delta_R, delta_T = PoseTool.Rt(delta_pose)
        delta_angle = np.arccos((np.trace(delta_R) - 1) / 2).item() * 180 / math.pi
        delta_translation = np.linalg.norm(delta_T).item()

        _DEBUG = False
        if _DEBUG:
            from utils.visualization import show_pcd
            src_init = init_SE3[:3, :3] @ src_pcd + init_SE3[:3, 3:]
            src_icp = icp_SE3[:3, :3] @ src_pcd + icp_SE3[:3, 3:]
            show_pcd([src_pcd.T, dst_pcd.T], [[1, 0, 0], [0, 1, 0]], window_name='origin')
            show_pcd([src_init.T, dst_pcd.T], [[1, 0, 0], [0, 1, 0]], window_name='gt')
            show_pcd([src_icp.T, dst_pcd.T], [[1, 0, 0], [0, 1, 0]],
                     window_name=f'icp: delta_angle={delta_angle:.2f}, delta_translation={delta_translation:.2f}')

        if delta_angle > 5 or delta_translation > 2:
            SE3_refinement.append(init_SE3)
            logger.warning('A suspected failed icp refinement has been discarded')
        else:
            SE3_refinement.append(icp_SE3)
    SE3_refinement = np.stack(SE3_refinement, axis=0)
    SE3_refinement = torch.from_numpy(SE3_refinement).float().to(device)
    R, T = SE3_refinement[:, :3, :3], SE3_refinement[:, :3, 3:]
    return R, T
