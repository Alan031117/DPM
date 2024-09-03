import os
import json
import numpy as np
from typing import Literal


def parse_evo(evo_response: str) -> dict:
    """从evo的返回值中解析ape指标"""
    result = dict()
    for line in evo_response.splitlines():
        line_strip = line.strip()
        if line_strip.startswith("mean"):
            result['mean'] = f'{line.split()[1]:>10s}'
        elif line_strip.startswith("max"):
            result['max'] = f'{line.split()[1]:>10s}'
        elif line_strip.startswith("min"):
            result['min'] = f'{line.split()[1]:>10s}'
        elif line_strip.startswith("median"):
            result['median'] = f'{line.split()[1]:>10s}'
        elif line_strip.startswith("rmse"):
            result['rmse'] = f'{line.split()[1]:>10s}'
        elif line_strip.startswith("sse"):
            result['sse'] = f'{line.split()[1]:>10s}'
        elif line_strip.startswith("std"):
            result['std'] = f'{line.split()[1]:>10s}'
    return result


def eval_ape(gt_file_path: str, pred_file_path: str, step_file_path: str,
             dataset_type: Literal['kitti', 'kitti360', 'mulran'],
             calib_file_path: str = '', out_dir: str = '') -> dict:
    """
    评估APE指标

    :param gt_file_path: 真实位姿文件路径
    :param pred_file_path: 预测位姿文件路径
    :param step_file_path: 预测位姿的序号
    :param dataset_type: 数据集类型，支持kitti与kitti360的位姿组织方式
    :param calib_file_path: 标定矩阵文件路径
    :param out_dir: 评估结果输出目录
    :return: 指标
    """
    # 读取位姿
    with open(gt_file_path, 'r') as f:
        gt_pose = f.readlines()
        gt_pose = np.array(list(filter(lambda x: len(x) > 0, [i.strip().split() for i in gt_pose])), dtype=np.float)
    with open(pred_file_path, 'r') as f:
        pred_pose = f.readlines()
        pred_pose = np.array(list(filter(lambda x: len(x) > 0, [i.strip().split() for i in pred_pose])), dtype=np.float)
    with open(step_file_path, 'r') as f:
        pred_step = f.readlines()
        pred_step = np.array(list(filter(lambda x: len(x) > 0, [i.strip().split() for i in pred_step])), dtype=np.int)
        pred_step = pred_step.flatten()

    # 对齐预测位姿与gt位姿的序号
    if dataset_type == 'kitti360':
        # kitti360格式pose，第0位为对应的帧序号
        gt_step = gt_pose[:, 0].astype(np.int)
        gt_pose = gt_pose[:, 1:13]
        inter_step = np.intersect1d(gt_step, pred_step)
        gt_mask = np.zeros_like(gt_step).astype(np.bool)
        pred_mask = np.zeros_like(pred_step).astype(np.bool)
        gt_i, pred_i = 0, 0
        for i in inter_step:
            while gt_step[gt_i] != i:
                gt_i += 1
            gt_mask[gt_i] = True
            while pred_step[pred_i] != i:
                pred_i += 1
            pred_mask[pred_i] = True
        gt_pose = gt_pose[gt_mask]
        pred_pose = pred_pose[pred_mask]
    else:
        gt_pose = gt_pose[pred_step]

    padding = np.array([[0, 0, 0, 1]], dtype=np.float)
    gt_pose_se3 = np.concatenate([gt_pose, padding.repeat(gt_pose.shape[0], axis=0)], axis=1).reshape(-1, 4, 4)
    pred_pose_se3 = np.concatenate([pred_pose, padding.repeat(pred_pose.shape[0], axis=0)], axis=1).reshape(-1, 4, 4)

    # 利用标定矩阵变换坐标系（如果需要）
    if calib_file_path != '':
        with open(calib_file_path, 'r') as f:
            calib = f.readlines()
        if dataset_type == 'kitti':
            calib = np.array(calib[-1][3:].strip().split(), dtype=np.float)
            calib = np.concatenate([calib.reshape(-1, 4), padding])
            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(calib), gt_pose_se3)
            gt_pose_se3 = np.einsum("...ij,...jk->...ik", left, calib)
        elif dataset_type == 'kitti360':
            calib = np.array(list(filter(lambda x: len(x) > 0, [i.strip().split() for i in calib])), dtype=np.float)
            calib = np.concatenate([calib.reshape(-1, 4), padding])
            gt_pose_se3 = gt_pose_se3 @ np.linalg.inv(calib)[np.newaxis, :, :]
        else:
            raise ValueError

    # 首帧对齐
    pred_pose_se3 = (gt_pose_se3[:1] @ np.linalg.inv(pred_pose_se3[:1])) @ pred_pose_se3

    gt_pose = gt_pose_se3.reshape(-1, 16)[:, :12]
    pred_pose = pred_pose_se3.reshape(-1, 16)[:, :12]
    temp_gt_file_path = os.path.join(os.path.dirname(pred_file_path), '__temp_gt_file.txt')
    temp_pred_file_path = os.path.join(os.path.dirname(pred_file_path), '__temp_pred_file.txt')
    with open(temp_gt_file_path, 'w+') as f:
        for pose in gt_pose.tolist():
            f.write(' '.join([f'{i:.10f}' for i in pose]) + '\n')
    with open(temp_pred_file_path, 'w+') as f:
        for pose in pred_pose.tolist():
            f.write(' '.join([f'{i:.10f}' for i in pose]) + '\n')

    # 使用evo计算ape
    if out_dir == '':
        out_dir = os.path.dirname(pred_file_path)
    save_plot = os.path.join(out_dir, 'ape.jpg')
    ape_map = os.path.join(out_dir, 'ape_map.jpg')
    ape_raw = os.path.join(out_dir, 'ape_raw.jpg')
    out_file = os.path.join(out_dir, 'ape.txt')
    for file_path in [ape_map, ape_raw, out_file]:
        if os.path.exists(file_path):
            os.remove(file_path)  # delete plot file if exist
    command = f'evo_ape kitti -a --plot_mode xy {temp_gt_file_path} {temp_pred_file_path} --save_plot {save_plot}'
    ape_result = parse_evo(os.popen(command).read())
    with open(out_file, 'w+') as f:
        f.write(json.dumps(ape_result, indent=4))

    # 删除临时文件
    os.remove(temp_gt_file_path)
    os.remove(temp_pred_file_path)
    return ape_result
