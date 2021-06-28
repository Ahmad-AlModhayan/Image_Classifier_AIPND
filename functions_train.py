import torch
from torchvision import transforms, datasets


####################################################
# Function of transforms
def train_data_load(train_dir):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    return train_loader

####################################################
# Function of transforms valid&test
def valid_test_data_load(test_dir):
    test_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    return test_loader

####################################################

