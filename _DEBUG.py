import numpy as np
from utils.visualization import show_pcd

# 定义用于显示不同类别点云的颜色列表，每个颜色对应一个类别
colors = [[1, 0, 0],  # 红色
          [0, 1, 0],  # 绿色
          [0, 0, 1],  # 蓝色
          [0, 1, 1],  # 青色
          [1, 0, 1],  # 品红色
          [1, 1, 0],  # 黄色
          [0, 0.5, 0.5],  # 暗青色
          [0.5, 0, 0.5],  # 暗紫色
          [0.5, 0.5, 0]]  # 橄榄色

# 指定要加载的npz文件路径
npz_file = r"C:\Deeppointmap file\KITTI-mini\07\0.npz"

# 使用numpy的np.load函数加载npz文件，并允许文件内对象被pickle序列化
with np.load(npz_file, allow_pickle=True) as npz:
    # 从npz文件中读取点云数据xyz，对应键为'lidar_pcd'
    # 这些数据的形状为(N, 3)，表示N个点的三维坐标，数据类型为f32（32位浮点数）
    xyz = npz['lidar_pcd']

    # 从npz文件中读取每个点的标签，对应键为'labels'
    label = npz['labels']

# 获取标签的唯一值集合，即点云中的所有类别
seg_label_set = np.unique(label)

# 初始化两个列表，用于存储每个类别的点云和对应的颜色
pcd_per_cls = []
color_per_cls = []

# 遍历每个唯一类别标签
for cls in seg_label_set:
    # 筛选出当前类别的点云数据
    cur_pcd = xyz[label == cls]

    # 根据类别选择对应的颜色，如果类别数量超过颜色列表的长度，则通过取模循环使用颜色
    cur_color = colors[cls % len(colors)]

    # 将当前类别的点云数据和颜色添加到列表中
    pcd_per_cls.append(cur_pcd)
    color_per_cls.append(cur_color)

# 调用show_pcd函数显示点云数据，使用为每个类别指定的颜色
show_pcd(pcd_per_cls, color_per_cls)
