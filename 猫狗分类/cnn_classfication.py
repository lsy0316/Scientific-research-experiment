import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time
import os

def train_model(model, model_name, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # 只训练未冻结的参数（分类头）
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses = []
    val_accs = []
    best_acc = 0.0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        val_accs.append(acc)

        print(f'Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')

        scheduler.step()

    return train_losses, val_accs, best_acc

def get_model(model_name='vgg'):
    if model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnet':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'vit':
        model = models.vit_b_16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, 2)
    else:
        raise ValueError('model_name must be vgg, resnet or vit')
    return model

def main():
    print("当前工作目录:", os.getcwd())
    print("cat_dog 存在?", os.path.exists('./cat_dog'))
    print("cat 子目录存在?", os.path.exists('./cat_dog/cat'))
    print("dog 子目录存在?", os.path.exists('./cat_dog/dog'))

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root='./cat_dog', transform=transform_train)
    print(f"数据总量: {len(full_dataset)} 张图片")
    print(f"类别: {full_dataset.classes}")

    # 划分训练集和验证集 (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = transform_val

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    models_to_train = ['vgg', 'resnet', 'vit']
    results = {}

    for model_name in models_to_train:
        print(f'\n===== 开始训练 {model_name.upper()} =====')
        model = get_model(model_name)
        start_time = time.time()
        losses, accs, best_acc = train_model(model, model_name, train_loader, val_loader, epochs=10, lr=0.001)
        elapsed = time.time() - start_time
        results[model_name] = {
            'losses': losses,
            'accs': accs,
            'time': elapsed,
            'best_acc': best_acc
        }
        print(f'{model_name.upper()} 训练完成，耗时 {elapsed:.2f}s，最佳验证准确率: {best_acc:.4f}')

    plt.figure(figsize=(12, 4))
    for i, name in enumerate(models_to_train):
        plt.subplot(1, 3, i+1)
        plt.plot(results[name]['losses'], label='Train Loss', marker='o')
        plt.plot(results[name]['accs'], label='Val Acc', marker='s')
        plt.title(name.upper())
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    print("\n" + "="*50)
    print("最终性能对比")
    print("="*50)
    print(f"{'模型':<10} {'最佳验证准确率':<18} {'训练时间(10轮)':<15}")
    print("-"*50)
    for name in models_to_train:
        print(f"{name.upper():<10} {results[name]['best_acc']:.4f}               {results[name]['time']:.2f} s")

if __name__ == '__main__':
    main()
