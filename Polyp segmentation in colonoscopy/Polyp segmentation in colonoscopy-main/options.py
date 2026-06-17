import argparse

parser = argparse.ArgumentParser()
# training
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batchsize', type=int, default=24)
parser.add_argument('--trainsize', type=int, default=352)
parser.add_argument('--clip', type=float, default=0.5)
parser.add_argument('--decay_rate', type=float, default=0.2)
parser.add_argument('--decay_epoch', type=int, default=40)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--backbone', type=str, default='vmamba')
parser.add_argument('--vmamba_cfg', type=str, default='tiny', choices=['tiny','small','base'])
parser.add_argument('--vmamba_pretrain', type=str, default='pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth')

parser.add_argument('--rgb_root', type=str, default='./data/TrainDatasetEdges_with_depths/images')
parser.add_argument('--depth_root', type=str, default='./data/TrainDatasetEdges_with_depths')
parser.add_argument('--gt_root', type=str, default='./data/TrainDatasetEdges_with_depths/masks')
parser.add_argument('--edge_root', type=str, default='./data/TrainDatasetEdges_with_depths/edges')
parser.add_argument('--depth_folder', type=str, default='depthsv2_vitl/gray')  
parser.add_argument('--save_path', type=str, default='./CIRNet_cpts')
# loss
parser.add_argument('--edge_weight', type=float, default=0)
# test
parser.add_argument('--testsize', type=int, default=352)
parser.add_argument('--test_path', type=str, default='./data/TestDataset_with_depths')
parser.add_argument('--test_model', type=str, default='CIRNet_VMamba.pth')

opt = parser.parse_args()