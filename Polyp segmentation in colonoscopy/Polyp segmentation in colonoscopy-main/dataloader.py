import os
import random
import struct
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def randomFlip(img, depth, gt, edge):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, depth, gt, edge

def randomRotation(img, depth, gt, edge):
    if random.random() > 0.8:
        angle = np.random.randint(-15, 15)
        img = img.rotate(angle, Image.BICUBIC)
        depth = depth.rotate(angle, Image.BICUBIC)
        gt = gt.rotate(angle, Image.NEAREST)
        edge = edge.rotate(angle, Image.NEAREST)
    return img, depth, gt, edge

def small_object_focus_crop(img, depth, gt, edge, base_size=352):
    """随机裁剪息肉区域并放大，强制模型关注小目标"""
    if random.random() < 0.5:
        # 找到GT非零区域（息肉区域）
        img_np = np.array(gt)
        y, x = np.where(img_np > 0)
        if len(x) > 0:
            cx = int(np.mean(x))
            cy = int(np.mean(y))
            # 随机裁剪尺寸 96~160
            crop_size = random.randint(96, 160)
            left = max(cx - crop_size//2, 0)
            top = max(cy - crop_size//2, 0)
            right = min(left + crop_size, img.width)
            bottom = min(top + crop_size, img.height)
            # 裁剪
            img = img.crop((left, top, right, bottom))
            depth = depth.crop((left, top, right, bottom))
            gt = gt.crop((left, top, right, bottom))
            edge = edge.crop((left, top, right, bottom))
            # 放大到 base_size
            img = img.resize((base_size, base_size), Image.BICUBIC)
            depth = depth.resize((base_size, base_size), Image.BICUBIC)
            gt = gt.resize((base_size, base_size), Image.NEAREST)
            edge = edge.resize((base_size, base_size), Image.NEAREST)
            return img, depth, gt, edge
    # 否则做常规缩放
    img = img.resize((base_size, base_size), Image.BICUBIC)
    depth = depth.resize((base_size, base_size), Image.BICUBIC)
    gt = gt.resize((base_size, base_size), Image.NEAREST)
    edge = edge.resize((base_size, base_size), Image.NEAREST)
    return img, depth, gt, edge

def colorJitter(img):
    if random.random() > 0.5:
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        hue = random.uniform(-0.1, 0.1)
        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)
        img = TF.adjust_saturation(img, saturation)
        img = TF.adjust_hue(img, hue)
    return img

def gaussianBlur(img):
    if random.random() > 0.5:
        img = TF.gaussian_blur(img, kernel_size=3, sigma=(0.1, 1.0))
    return img


class PolypDataset(data.Dataset):
    def __init__(self, image_root, depth_root, gt_root, edge_root, trainsize=352, depth_folder='depthsv2_vitl/gray'):
        self.trainsize = trainsize
        self.depth_folder = depth_folder
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(('.jpg','.png'))])
        depth_dir = os.path.join(depth_root, depth_folder)
        self.depths = sorted([os.path.join(depth_dir, f) for f in os.listdir(image_root) if f.endswith(('.jpg','.png'))])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith(('.jpg','.png'))])
        self.edges = sorted([os.path.join(edge_root, f) for f in os.listdir(edge_root) if f.endswith(('.jpg','.png'))])
        self.filter_files()
        self.size = len(self.images)

    def filter_files(self):
        valid = []
        for i in range(len(self.images)):
            try:
                img = Image.open(self.images[i])
                img.load()
                gt = Image.open(self.gts[i])
                gt.load()
                depth = Image.open(self.depths[i])
                depth.load()
                edge = Image.open(self.edges[i])
                edge.load()
                if img.size == gt.size:
                    valid.append(i)
            except Exception as e:
                print(f"Skip corrupted: {self.images[i]} - {e}")
                continue
        self.images = [self.images[i] for i in valid]
        self.depths = [self.depths[i] for i in valid]
        self.gts = [self.gts[i] for i in valid]
        self.edges = [self.edges[i] for i in valid]
        print(f"Valid samples: {len(self.images)}")

    def __getitem__(self, index):
        for attempt in range(10):
            try:
                image = self.rgb_loader(self.images[index])
                depth = self.binary_loader(self.depths[index])
                gt = self.binary_loader(self.gts[index])
                edge = self.binary_loader(self.edges[index])

                # Basic geometry
                image, depth, gt, edge = randomFlip(image, depth, gt, edge)
                image, depth, gt, edge = randomRotation(image, depth, gt, edge)

                # Small object focus crop (critical for ETIS)
                image, depth, gt, edge = small_object_focus_crop(image, depth, gt, edge, self.trainsize)

                # Color & blur
                image = colorJitter(image)
                image = gaussianBlur(image)

                # To tensor & normalize
                img_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
                ])
                depth_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])
                gt_transform = transforms.ToTensor()
                edge_transform = transforms.ToTensor()

                image = img_transform(image)
                depth = depth_transform(depth)
                gt = gt_transform(gt)
                edge = edge_transform(edge)

                return image, depth, gt, edge
            except (OSError, IOError, struct.error) as e:
                print(f"Fail load {index} ({attempt+1}/10): {e}")
                index = random.randint(0, self.size-1)
        raise RuntimeError("Cannot load sample after retries")

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(image_root, depth_root, gt_root, edge_root, batchsize, trainsize=352,
               shuffle=True, num_workers=8, pin_memory=True, depth_folder='depthsv2_vitl/gray'):
    dataset = PolypDataset(image_root, depth_root, gt_root, edge_root, trainsize, depth_folder)
    loader = data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory)
    return loader


class test_rgbd_dataset:
    def __init__(self, image_root, depth_root, gt_root, testsize=352, depth_folder='depthsv2_vitl/gray'):
        self.testsize = testsize
        self.depth_folder = depth_folder
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(('.jpg','.png'))])
        depth_dir = os.path.join(depth_root, depth_folder)
        self.depths = sorted([os.path.join(depth_dir, f) for f in os.listdir(image_root) if f.endswith(('.jpg','.png'))])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith(('.jpg','.png'))])
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.gt_transform = transforms.ToTensor()
        self.index = 0

    def filter_files(self):
        valid = []
        for i in range(len(self.images)):
            try:
                img = Image.open(self.images[i])
                img.load()
                gt = Image.open(self.gts[i])
                gt.load()
                depth = Image.open(self.depths[i])
                depth.load()
                if img.size == gt.size:
                    valid.append(i)
            except Exception as e:
                print(f"Test skip: {self.images[i]} - {e}")
                continue
        self.images = [self.images[i] for i in valid]
        self.depths = [self.depths[i] for i in valid]
        self.gts = [self.gts[i] for i in valid]

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depth_transform(depth).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = self.gt_transform(gt).unsqueeze(0)
        name = os.path.basename(self.images[self.index])
        self.index = (self.index + 1) % self.size
        return image, depth, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size