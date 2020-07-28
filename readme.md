# 基于Mnist，DeepFool攻击ResNet18

## 文件说明

- `data`：Mnist数据集
- `images`：对抗样本，9957张，剩余43张图像网络本身无法正确分类，为原图像
- `model`：ResNet18的训练函数，和预训练参数文件
- `logs`：训练日志

## 代码文档说明

- `model.py`：ResNet18网络定义
- `utils.py`：其他函数
- `deepfool_fashion.py`：DeepFool核心算法，当使用其他数据集时，代码中的图像预处理需要微调
- `Resnet-Deepfool.ipynd`：主函数
- `tran.py`：模型训练函数。

## 环境配置

- python 3.7
- pytorch 1.4.0
- cuda 10.1
