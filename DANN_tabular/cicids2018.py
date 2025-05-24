import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch
import params
from torch.utils.data import random_split

transform = {
    'train': transforms.Compose([
        transforms.ToTensor(),  # 轉為 Tensor
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ])
}

# 資料集路徑
dataset_dir_2018 = 'data/image/cicids2018/train_224/'

# 使用 ImageFolder 自動根據子資料夾讀取標籤
dataset_2018 = datasets.ImageFolder(root=dataset_dir_2018, transform=transform['train'])

#切割出train, valid, test三個dataset，比例0.7:0.15:0.15
tatal_size = len(dataset_2018)
train_size = int(0.7 * tatal_size)
valid_size = int(0.15 * tatal_size)
test_size = tatal_size - train_size - valid_size
train_dataset_2018, valid_dataset_2018, test_dataset_2018 = random_split(dataset_2018, [train_size, valid_size, test_size])

#創建train, valid, test三個dataloader
train_loader_2018 = DataLoader(
    train_dataset_2018,
    batch_size=params.batch_size,
    # sampler=train_sampler,
    shuffle=True,
    num_workers=params.num_workers
)

valid_loader_2018 = DataLoader(
    valid_dataset_2018,
    batch_size=params.batch_size,
    # sampler=valid_sampler,
    shuffle=False,
    num_workers=params.num_workers
)

test_loader_2018 = DataLoader(
    test_dataset_2018,
    batch_size=params.batch_size,
    shuffle=False,
    num_workers=params.num_workers
)


def one_hot_embedding(labels, num_classes=10):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]
