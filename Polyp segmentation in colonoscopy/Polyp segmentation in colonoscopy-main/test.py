import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from options import opt
from model.CIRNet_VMamba import CIRNet_VMamba
from dataloader import test_rgbd_dataset
from utils import dice_iou, cal_mae, cal_em, cal_sm, cal_wfm

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def find_latest_model(model_dir='./CIRNet_cpts', pattern='CIRNet_VMamba_epoch_*.pth'):
    files = glob.glob(os.path.join(model_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No model found in {model_dir}")
    latest = max(files, key=os.path.getctime)
    print(f"Using model: {latest}")
    return latest

if opt.test_model == 'CIRNet_VMamba.pth' and not os.path.exists(opt.test_model):
    opt.test_model = find_latest_model()
elif not os.path.exists(opt.test_model):
    raise FileNotFoundError(f"Model {opt.test_model} not found.")

model = CIRNet_VMamba(vmamba_cfg=opt.vmamba_cfg, pretrained_path=None)
state = torch.load(opt.test_model, map_location='cpu')
model.load_state_dict(state)
model.cuda()
model.eval()

test_datasets = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
dataset_path = opt.test_path

results = {}
for ds in test_datasets:
    print(f"\nTesting on {ds}...")
    img_root = os.path.join(dataset_path, ds, 'images')
    depth_root = os.path.join(dataset_path, ds)
    gt_root = os.path.join(dataset_path, ds, 'masks')
    loader = test_rgbd_dataset(img_root, depth_root, gt_root, opt.testsize, depth_folder='depthsv2_vitl/gray')

    metrics = {'dice':[], 'iou':[], 'wfm':[], 'sm':[], 'em':[], 'mae':[]}
    for i in range(loader.size):
        image, depth, gt, name = loader.load_data()
        gt_np = gt.squeeze().cpu().numpy()
        with torch.no_grad():
            _, _, pred, _, _, _ = model(image.cuda(), depth.cuda())
            pred = torch.sigmoid(pred)
            pred = F.interpolate(pred, size=gt_np.shape, mode='bilinear', align_corners=False)
            pred_np = pred.squeeze().cpu().numpy()
            pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)

        dice, iou = dice_iou(pred_np, gt_np)
        mae = cal_mae(pred_np, gt_np)
        em = cal_em(pred_np, gt_np)
        sm = cal_sm(pred_np, gt_np)
        wfm = cal_wfm(pred_np, gt_np)

        metrics['dice'].append(dice); metrics['iou'].append(iou); metrics['wfm'].append(wfm)
        metrics['sm'].append(sm); metrics['em'].append(em); metrics['mae'].append(mae)

        os.makedirs(f'./test_maps/{ds}', exist_ok=True)
        cv2.imwrite(f'./test_maps/{ds}/{name}', (pred_np*255).astype(np.uint8))

    avg = {k: np.mean(v) for k,v in metrics.items()}
    results[ds] = avg
    print(f"{ds}: Dice={avg['dice']:.4f}, IoU={avg['iou']:.4f}, WFM={avg['wfm']:.4f}, S={avg['sm']:.4f}, E={avg['em']:.4f}, MAE={avg['mae']:.4f}")

print("\n===== Final Results =====")
for ds, m in results.items():
    print(f"{ds:20s} Dice {m['dice']:.4f}  IoU {m['iou']:.4f}  WFM {m['wfm']:.4f}  S-measure {m['sm']:.4f}  E-measure {m['em']:.4f}  MAE {m['mae']:.4f}")