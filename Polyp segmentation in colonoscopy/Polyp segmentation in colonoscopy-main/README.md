# CIRNet-VMamba 用于息肉分割

本项目基于 CIRNet 论文，将骨干网络从 ResNet 替换为 VMamba，用于息肉分割任务。

## 参考论文

Runmin Cong, Qinwei Lin, Chen Zhang, Chongyi Li, Xiaochun Cao, Qingming Huang, and Yao Zhao, CIR-Net: Cross-modality interaction and refinement for RGB-D salient object detection, IEEE Transactions on Image Processing, vol. 31, pp. 6800-6815, 2022.

## 项目概述

本实现将 CIR-Net 架构适配到医学图像分割领域，专门用于结肠镜检查中的息肉分割。主要修改是将原始的 ResNet 骨干网络替换为 VMamba，利用状态空间模型在医学图像场景中实现更好的特征提取。

## Pytorch 实现

* 基于 VMamba 骨干网络的 CIR-Net Pytorch 实现
* 预训练模型：
  - 我们提供测试代码。如果您想测试我们的模型，请下载预训练模型，解压后将其放入相应的文件夹
  - 使用 VMamba 骨干网络的息肉分割预训练模型：[下载链接] (密码:1234)

## 环境要求

* Python 3.7+
* torch>=1.10.1
* torchvision>=0.11.2
* opencv-python
* Pillow
* timm (用于 VMamba 实现)

## 数据准备

* 请下载息肉分割训练数据并将其放入 `data` 文件夹。
* 数据集应包含：
  - 训练图像及其对应的标注
  - 测试图像及其对应的标注
* 常用的息肉分割数据集包括 CVC-ClinicDB、ETIS-Larib、Kvasir-SEG 等

## 测试
```
python3 CIRNet_test.py --backbone VMamba --test_model CIRNet_VMamba.pth
```

## 训练
```
python3 CIRNet_train.py --backbone VMamba
```

* 您可以在 `test_maps` 文件夹中找到结果

## 引用

如果您使用我们的 CIR-Net，请引用我们的论文：

     @article{crm/tip22/CIRNet,
       title={{CIR-Net}: Cross-modality interaction and refinement for {RGB-D} salient object detection},
       author={Cong, Runmin and Lin, Qinwei and Zhang, Chen and Li, Chongyi and Cao, Xiaochun and Huang, Qingming and Zhao, Yao },
       journal={IEEE Trans. Image Process. },
       volume={31},
       pages={6800-6815},
       year={2022},
      }

## 联系我们
如果您有任何问题，请联系 Runmin Cong (rmcong@bjtu.edu.cn) 或 Qinwei Lin (lqw22@mails.tsinghua.edu.cn)。