import os.path as op

CIFAR100_NAME = 'CIFAR-100'
IMAGENET_NAME = 'ImageNet'

TITLE_FONT_SIZE = 16
AXES_FONT_SIZE = 18

EXAMPLES_PALETTE = {
    'most memorized': 'r',
    'least memorized': 'b',
    'random': 'g'
}

DS_BINS_NUM = {CIFAR100_NAME: 300, IMAGENET_NAME: 1000}


def find_dataset_name(some_path):
    user_abs_path = op.expanduser(some_path)
    paths = user_abs_path.split('/')
    if paths[6] == 'mem_cifar100':
        return CIFAR100_NAME
    elif paths[6] == 'mem_imagenet':
        return IMAGENET_NAME
    else:
        return None


def layer_ds_title(ds, layer_index):
    return "%s, Stage %d" % (ds, layer_index + 1)



