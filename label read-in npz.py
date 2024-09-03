import os
import numpy as np

# 定义文件路径
label_dir = r'C:\Deeppointmap file\semantic-kitti\dataset\sequences\00\labels'  # .label 文件目录
npz_dir = r'C:\Deeppointmap file\KITTI-mini\07\0'  # .npz 文件目录
output_dir = r'C:\Users\HUAWEI\Desktop\npz'  # 更新后的.npz文件保存目录

label_files = os.listdir(label_dir)

for label_file in label_files:
    if label_file.endswith('.label'):
        sample_id = os.path.splitext(label_file)[0]
        npz_file = os.path.join(npz_dir, f'{int(sample_id)}.npz')
        label_path = os.path.join(label_dir, label_file)
        labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
        npz_data = np.load(npz_file)
        data_dict = {key: npz_data[key] for key in npz_data.files}
        data_dict['lidar_seg'] = labels.astype(np.int64)  # 添加新的标签数据

        npz_data.close()

        output_npz_file = os.path.join(output_dir, f'{int(sample_id)}.npz')
        np.savez(output_npz_file, **data_dict)

