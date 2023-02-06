import os.path as op
import torch
from torchvision.transforms import transforms

import base_settings


def get_train_transform(data_augmentation, dataset, normalize_data=True, noise_level=0.0):
    assert dataset in {'cifar10', 'cifar100', 'imagenet', 'mini_imagenet'}

    if data_augmentation:
        if dataset == 'cifar10' or dataset == 'cifar100':
            random_crop = transforms.RandomCrop(base_settings.CIFAR_RANDOM_CROP_SIZE,
                                                padding=base_settings.CIFAR_RANDOM_CROP_PADDING)
        else:
            random_crop = transforms.RandomCrop(base_settings.MINI_IMAGENET_RANDOM_CROP_SIZE,
                                                padding=base_settings.MINI_IMAGENET_RANDOM_CROP_PADDING)
        train_transform = [random_crop, transforms.RandomHorizontalFlip()]
    else:
        train_transform = []

    if dataset == 'cifar10':
        normalize_transform = transforms.Normalize(base_settings.CIFAR10_MEAN, base_settings.CIFAR10_STD)
    elif dataset == 'cifar100':
        normalize_transform = transforms.Normalize(base_settings.CIFAR100_MEAN, base_settings.CIFAR100_STD)
    else:
        normalize_transform = transforms.Normalize(base_settings.MINI_IMAGENET_MEAN, base_settings.MINI_IMAGENET_STD)

    if normalize_data:
        train_transform.extend([transforms.ToTensor(), normalize_transform])
    else:
        train_transform.extend([transforms.ToTensor(), transforms.Lambda(lambda t: t.mul(255.))])

    if noise_level > 0.0:
        noise_transform = lambda x : x + torch.randn_like(x)*noise_level
        train_transform.extend([noise_transform])

    return transforms.Compose(train_transform)


def get_test_transform(dataset, normalize_data=True, noise_level=0.0):
    assert dataset in {'cifar10', 'cifar100', 'imagenet', 'mini_imagenet'} or op.exists(dataset)

    if dataset == 'cifar10':
        normalize_transform = transforms.Normalize(base_settings.CIFAR10_MEAN, base_settings.CIFAR10_STD)
    elif dataset == 'cifar100':
        normalize_transform = transforms.Normalize(base_settings.CIFAR100_MEAN, base_settings.CIFAR100_STD)
    elif dataset in {'imagenet', 'mini_imagenet'}:
        normalize_transform = transforms.Normalize(base_settings.MINI_IMAGENET_MEAN, base_settings.MINI_IMAGENET_STD)

    test_transform = []

    if normalize_data:
        test_transform.extend([transforms.ToTensor(), normalize_transform])
    else:
        test_transform.extend([transforms.ToTensor(), transforms.Lambda(lambda t: t.mul(255.))])

    if noise_level > 0.0:
        noise_transform = lambda x : x + torch.randn_like(x)*noise_level
        test_transform.extend([noise_transform])

    return transforms.Compose(test_transform)
