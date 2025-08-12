from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

import os


cifar_mean = [0.4914, 0.4822, 0.4465]
cifar_std = [0.2023, 0.1994, 0.2010]

general_mean = [0.5, 0.5, 0.5]
general_std = [0.5, 0.5, 0.5]

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

cifar_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

vgg_train_transfrom = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

imagenet_train_transfrom = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

cifar_test_transform = transforms.Compose([
    transforms.Resize(size = (32,32)),
    transforms.ToTensor(),
    ])

vgg_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

imagenet_test_transfrom = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),])

def normalize(t, mean, std):
    norm_t = torch.zeros_like(t)
    norm_t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    norm_t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    norm_t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return norm_t


def denormalize(t, mean, std):
    denorm_t = torch.zeros_like(t)
    denorm_t[:, 0, :, :] = t[:, 0, :, :] * std[0] + mean[0]
    denorm_t[:, 1, :, :] = t[:, 1, :, :] * std[1] + mean[1]
    denorm_t[:, 2, :, :] = t[:, 2, :, :] * std[2] + mean[2]
    return denorm_t
    
    
def data_process(dataset='cifar10', 
                 data_path="./data",
                 batch_size=64):
    if dataset=='cifar10':
        train_set = datasets.CIFAR10(root=data_path, train=True, download=True, transform=cifar_train_transform)
        test_set = datasets.CIFAR10(root=data_path, train=False, download=True, transform=cifar_test_transform )
        mean = cifar_mean
        std = cifar_std
        img_size = (32, 32)
        class_names = train_set.classes
    elif dataset=='imagenette':
        train_set = datasets.ImageFolder(root=os.path.join(data_path, 'imagenette2/train'), transform=imagenet_train_transfrom)
        test_set = datasets.ImageFolder(root=os.path.join(data_path, 'imagenette2/val'), transform=imagenet_test_transfrom)
        mean = imagenet_mean
        std = imagenet_std
        img_size = (224, 224)
        class_names = train_set.classes
    elif dataset=='imagenet':
        train_set = datasets.ImageFolder(root=os.path.join(data_path, 'imagenet_subtrain'), transform=imagenet_train_transfrom)
        # UU
        test_set = datasets.ImageFolder(root=os.path.join(data_path, 'imagenet_subval'), transform=imagenet_test_transfrom)
        mean = imagenet_mean
        std = imagenet_std
        img_size = (224, 224)
        class_names = train_set.classes
    elif dataset=='vggface':
        train_set = datasets.ImageFolder(root=os.path.join(data_path, 'vggface/train'), transform=vgg_train_transfrom)
        test_set = datasets.ImageFolder(root=os.path.join(data_path, 'vggface/test'), transform=vgg_test_transform)
        mean = general_mean
        std = general_std
        img_size = (224, 224)
        class_names = train_set.classes
    elif dataset=='vggface2':
        train_set = datasets.ImageFolder(root=os.path.join(data_path, 'vggface2/train'), transform=vgg_train_transfrom)
        test_set = datasets.ImageFolder(root=os.path.join(data_path, 'vggface2/test'), transform=vgg_test_transform)
        mean = general_mean
        std = general_std
        img_size = (224, 224)
        class_names = train_set.classes
    elif dataset == 'gtsrb':
        train_set = datasets.GTSRB(root=data_path, split='train', transform=cifar_train_transform, download=True)
        test_set = datasets.GTSRB(root=data_path, split='test', transform=cifar_test_transform, download=True)
        mean = general_mean
        std = general_std
        img_size = (32, 32)
        class_names = [
            'Speed limit (20km/h)',     # 0
            'Speed limit (30km/h)',     # 1
            'Speed limit (50km/h)',     # 2
            'Speed limit (60km/h)',     # 3
            'Speed limit (70km/h)',     # 4
            'Speed limit (80km/h)',     # 5
            'End of speed limit (80km/h)',  # 6
            'Speed limit (100km/h)',    # 7
            'Speed limit (120km/h)',    # 8
            'No passing',               # 9
            'No passing for vehicles over 3.5 metric tons',     # 10
            'Right-of-way at the next intersection',            # 11
            'Priority road',            # 12
            'Yield',                    # 13
            'Stop',                     # 14
            'No vehicles',              # 15    
            'Vehicles over 3.5 metric tons prohibited',         # 16 
            'No entry',                 # 17
            'General caution',          # 18
            'Dangerous curve to the left',            # 19
            'Dangerous curve to the right',      # 20
            'Double curve',             # 21
            'Bumpy road',               # 22
            'Slippery road',            # 23
            'Road narrows on the right',    # 24
            'Road work',                # 25
            'Traffic signals',          # 26
            'Pedestrians',              # 27
            'Children crossing',        # 28
            'Bicycles crossing',            # 29
            'Beware of ice/snow',        # 30
            'Wild animals crossing',     # 31
            'End of all speed and passing limits',              # 32 
            'Turn right ahead',         # 33
            'Turn left ahead',          # 34
            'Ahead only',               # 35
            'Go straight or right',     # 36    
            'Go straight or left',      # 37
            'Keep right',               # 38 
            'Keep left',                # 39 
            'Roundabout mandatory',     # 40
            'End of no passing',        # 41
            'End of no passing by vehicles over 3.5 metric tons']    # 42  
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_set), 'test': len(test_set)}
    # print(class_names)
    print(dataset_sizes)
    return dataloaders, dataset_sizes, len(class_names), mean, std, img_size