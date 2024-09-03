import torch
import torch.nn as nn
from torch import Tensor as Tensor


def CoarsePairingHead(emb_dim: int):
    # 粗匹配头,使输入特征更容易匹配
    return nn.Sequential(
        nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1),  # 一维卷积层，保持特征维度不变
        nn.ReLU(inplace=True),  # ReLU激活函数
        nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1)  # 再次使用一维卷积层
    )


def SimilarityHead(emb_dim: int):
    # 创建一个相似性头部网络，用于计算特征之间的相似性
    return nn.Sequential(
        nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1),  # 一维卷积层，保持特征维度不变
        nn.ReLU(inplace=True),  # ReLU激活函数
        nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1)  # 再次使用一维卷积层
    )


class OffsetHead(nn.Module):
    """预测匹配点间的相对偏移量"""
    def __init__(self, emb_dim: int, coor_dim: int = 3):
        super().__init__()
        # 多层感知机（MLP）用于逐步减少特征维度
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim // 2, kernel_size=1),  # 第一层一维卷积层，减少特征维度为一半
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv1d(in_channels=emb_dim // 2, out_channels=emb_dim // 4, kernel_size=1),  # 第二层一维卷积层，继续减少特征维度为原来四分之一
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv1d(in_channels=emb_dim // 4, out_channels=emb_dim // 8, kernel_size=1),  # 第三层一维卷积层，减少特征维度为原来八分之一
        )
        # 下采样操作，用于匹配MLP后的特征维度
        self.downsample = nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim // 8, kernel_size=1)
        # 预测头，用于计算最终的偏移量
        self.head = nn.Conv1d(in_channels=emb_dim // 8, out_channels=coor_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)  # ReLU激活函数

    def forward(self, pcd_fea: Tensor) -> Tensor:
        """
        :param pcd_fea: (B, C, N) 输入的点云特征
        :return: (B, coor_dim, N) 输出点云坐标的偏移量
        """
        out = self.mlp(pcd_fea)  # 将输入特征通过MLP处理
        identity = self.downsample(pcd_fea)  # 对输入特征进行下采样，作为跳跃连接的部分
        out = self.act(out + identity)  # 将处理后的特征与下采样特征相加，并通过激活函数处理
        out = self.head(out)  # 通过预测头生成最终的偏移量
        return out  # 返回计算出的偏移量


class LoopHead(nn.Module):
    """回环检测头，预测两帧之间是否产生回环"""
    def __init__(self, emb_dim: int):
        super().__init__()
        # 多层感知机（MLP），用于提取特征信息
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1),  # 一维卷积层，保持特征维度不变
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1)  # 再次使用一维卷积层
        )
        # 投影层，用于将特征映射到回环概率空间
        self.projection = nn.Sequential(
            nn.Linear(in_features=2 * emb_dim, out_features=2 * emb_dim),  # 全连接层，将特征映射到更高维度
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(in_features=2 * emb_dim, out_features=1),  # 全连接层，将特征映射到单一输出（回环概率）
            nn.Sigmoid()  # Sigmoid函数，用于将输出映射到[0,1]之间，表示回环概率
        )

    def forward(self, src_fea, dst_fea) -> Tensor:
        # 分别通过MLP提取源特征和目标特征
        src_fea = self.mlp(src_fea)
        dst_fea = self.mlp(dst_fea)

        # (B, C, N) -> (B, C)
        # 通过对每个特征点的特征进行均值池化，将特征的维度从(B, C, N)变为(B, C)
        src_fea = torch.mean(src_fea, dim=-1)
        dst_fea = torch.mean(dst_fea, dim=-1)

        # 将源特征和目标特征拼接，输入到投影层，得到回环检测概率
        loop_pro = self.projection(torch.cat([src_fea, dst_fea], dim=-1)).flatten()  # (B,)
        return loop_pro  # 返回回环检测的概率

