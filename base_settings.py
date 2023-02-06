import os.path as op
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# data settings
DATA_ROOT = "data"

CIFAR10_MEAN = 0.4914, 0.4822, 0.4465
CIFAR10_STD = 0.247, 0.243, 0.261
CIFAR10_CORRUPTED_LABELS_PATH = op.join(DATA_ROOT, 'cifar10_corrupted_labels.json')

CIFAR100_MEAN = 0.507, 0.487, 0.441
CIFAR100_STD = 0.267, 0.256, 0.276
CIFAR100_CORRUPTED_LABELS_PATH = op.join(DATA_ROOT, 'cifar100_corrupted_labels.json')

MINI_IMAGENET_ROOT = op.join(DATA_ROOT, 'mini-mini_imagenet')
MINI_IMAGENET_MEAN = 0.473, 0.449, 0.403
MINI_IMAGENET_STD = 0.277, 0.269, 0.282
MINI_IMAGENET_CORRUPTED_LABELS_PATH = op.join(MINI_IMAGENET_ROOT, 'corrupted_labels.json')

# experiments general settings
EXPERIMENT_ROOT = op.join(DATA_ROOT, "experiments")

DATA_AUGMENTATION = False

CIFAR_RANDOM_CROP_SIZE = 32
CIFAR_RANDOM_CROP_PADDING = 4
MINI_IMAGENET_RANDOM_CROP_SIZE = 84
MINI_IMAGENET_RANDOM_CROP_PADDING = 10


IMAGENET_DS_SETTINGS = {
    'batch_size': 64,
    'scale_img': '0_to_1',
    'ds_path': MINI_IMAGENET_ROOT
}

