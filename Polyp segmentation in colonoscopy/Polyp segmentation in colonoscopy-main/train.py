import os, time, random, logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from options import opt
from datetime import datetime
from tensorboardX import SummaryWriter

from model.CIRNet_VMamba import CIRNet_VMamba
from dataloader import get_loader
from utils import clip_gradient, adjust_lr

def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

seed_torch()

os.makedirs(opt.save_path, exist_ok=True)
logging.basicConfig(filename=os.path.join(opt.save_path, 'log.log'),
                    format='[%(asctime)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a')
logging.info("Optimized CIRNet-VMamba: small-object crop + edge fusion + DiceLoss")
logging.info(str(opt))

train_loader = get_loader(opt.rgb_root, opt.depth_root, opt.gt_root, opt.edge_root,
                          opt.batchsize, opt.trainsize, depth_folder=opt.depth_folder)
total_step = len(train_loader)

model = CIRNet_VMamba(vmamba_cfg=opt.vmamba_cfg, pretrained_path=opt.vmamba_pretrain)

print("\n" + "="*60)
print("Optimized features:")
print(" - Small object focus crop (zoom-in polyp)")
print(" - Edge feature modulation inside decoder")
print(" - Shallow feature injection (64-ch)")
print(" - Dice + BCE loss")
print("="*60 + "\n")

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.cuda()

if opt.load:
    model.load_state_dict(torch.load(opt.load))
    print(f"Loaded checkpoint {opt.load}")

optimizer = torch.optim.Adam(model.parameters(), opt.lr)
bce = torch.nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2. * intersection + smooth) / (union + smooth)

writer = SummaryWriter(os.path.join(opt.save_path, 'summary'))

def train(epoch):
    model.train()
    loss_avg = 0
    for i, (images, depths, gts, edges) in enumerate(train_loader, 1):
        images, depths, gts, edges = images.cuda(), depths.cuda(), gts.cuda(), edges.cuda()
        optimizer.zero_grad()
        s_rgb, s_depth, s_rgbd, s_edge1, s_edge2, s_edge3 = model(images, depths)

        gt_size = gts.shape[2:]
        s_rgb = F.interpolate(s_rgb, size=gt_size, mode='bilinear', align_corners=True)
        s_depth = F.interpolate(s_depth, size=gt_size, mode='bilinear', align_corners=True)
        s_rgbd = F.interpolate(s_rgbd, size=gt_size, mode='bilinear', align_corners=True)
        s_edge1 = F.interpolate(s_edge1, size=gt_size, mode='bilinear', align_corners=True)
        s_edge2 = F.interpolate(s_edge2, size=gt_size, mode='bilinear', align_corners=True)
        s_edge3 = F.interpolate(s_edge3, size=gt_size, mode='bilinear', align_corners=True)

        # Segmentation loss: BCE + Dice
        loss_r = bce(s_rgb, gts) + dice_loss(s_rgb, gts)
        loss_d = bce(s_depth, gts) + dice_loss(s_depth, gts)
        loss_rd = bce(s_rgbd, gts) + dice_loss(s_rgbd, gts)

        # Multi-scale edge loss
        loss_e = bce(s_edge1, edges) + 0.5 * bce(s_edge2, edges) + 0.3 * bce(s_edge3, edges)

        loss = loss_r + loss_d + loss_rd + opt.edge_weight * loss_e
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        loss_avg += loss_rd.item()
        if i % 50 == 0:
            print(f"{datetime.now()} Epoch [{epoch:03d}/{opt.epoch}] Step [{i:04d}/{total_step}] Loss: {loss_avg/i:.6f}")
            writer.add_scalar('Loss/step', loss_avg/i, (epoch-1)*total_step + i)

    avg_loss = loss_avg / total_step
    print(f"Epoch {epoch} finished, Avg Loss: {avg_loss:.6f}")
    logging.info(f"Epoch {epoch} Avg Loss: {avg_loss:.6f}")
    writer.add_scalar('Loss/epoch', avg_loss, epoch)

    if epoch > 60 and (epoch % 5 == 0 or epoch == opt.epoch):
        torch.save(model.state_dict(), os.path.join(opt.save_path, f'CIRNet_VMamba_epoch_{epoch}.pth'))

if __name__ == '__main__':
    print("Start training...")
    start_time = time.time()
    for epoch in range(1, opt.epoch+1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('lr', cur_lr, epoch)
        train(epoch)
        print(f"Time used: {(time.time()-start_time)/60:.2f} min")