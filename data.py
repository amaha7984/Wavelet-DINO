from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, DistributedSampler

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

def train_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

def val_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

def prepare_dataloader(path, transform, batch_size, is_train=True):
    dataset = ImageFolder(path, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=is_train)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
