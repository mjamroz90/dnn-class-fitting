from torch import nn
from torchvision import models


def resnet50(dataset, classes_num=None, norm=None):
    return resnet('resnet50', dataset, classes_num, norm)


def resnet18(dataset, classes_num=None, norm=None):
    return resnet('resnet18', dataset, classes_num, norm)


def resnet(net_type, dataset, classes_num=None, norm=None):
    assert net_type in {'resnet50', 'resnet18'}
    if net_type == 'resnet50':
        func_ = models.resnet50
    else:
        func_ = models.resnet18

    if norm is not None and norm == 'group_norm':
        norm_layer = lambda num_channels: nn.GroupNorm(num_groups=16, num_channels=num_channels)
    else:
        norm_layer = None

    if dataset == 'cifar100' or dataset == 'mini_imagenet':
        net = func_(pretrained=False, num_classes=100, zero_init_residual=True, norm_layer=norm_layer)

        if dataset == 'cifar100':
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Original implementation has nn.AdaptiveAvgPool2d((1, 1)) here, but it
            # fails with 'Output size is too small' in 'SpatialAveragePooling.cu'
            net.avgpool = nn.AvgPool2d((4, 4))
        else:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=6, bias=False)
            net.avgpool = nn.AvgPool2d((4, 4))

        net.maxpool = nn.Identity()
    else:
        net = func_(pretrained=False, num_classes=classes_num, zero_init_residual=True, norm_layer=norm_layer)

    return net
