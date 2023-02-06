import argparse
import os
import os.path as op

from tqdm import tqdm
import torch
import numpy as np
from torchvision.datasets import CIFAR100

from core import Smooth
from datasets.mini_imagenet import MiniImageNet
from datasets.rlcifar import CIFAR100IncludedIndices
from src.transforms import get_test_transform
from utils import fs_utils
import base_settings

N = 10**5
N0 = 100
ALPHA = 0.001


def get_dataset(dataset, test_transform, evaluate_on_test_set, val_indices):
    if evaluate_on_test_set and val_indices is not None:
        raise ValueError("--evaluate_on_test_set flag implies that val_indices argument should be None")

    if not evaluate_on_test_set:
        if dataset == 'cifar100':
            val_dataset = CIFAR100IncludedIndices(included_indices=val_indices, root=base_settings.DATA_ROOT,
                                                  download=False, train=True, transform=test_transform)
        else:
            val_dataset = MiniImageNet(train=True, transform=test_transform, indices=val_indices, reduce_dim=False)
    else:
        if dataset == 'cifar100':
            val_dataset = CIFAR100(root=base_settings.DATA_ROOT, download=True, train=False, transform=test_transform)
        else:
            val_dataset = MiniImageNet(train=True, transform=test_transform, reduce_dim=False)

    return val_dataset


def run_certification(dataset, index_range, model, sigma, batch_size, device):
    smoother = Smooth(model, num_classes=100, sigma=sigma, device=device)
    start_index, end_index = index_range
    result = []
    for i in tqdm(range(start_index, end_index, 1)):
        i_img = dataset[i][0].to(device)
        (pred, radius) = smoother.certify(i_img, n=N, n0=N0, alpha=ALPHA, batch_size=batch_size)

        result.append([pred, radius])

    return np.array(result, dtype=np.float32)


def prepare_index_range(ds_size, rank_index, world_size, index_range):
    if index_range is not None:
        global_start, global_stop = index_range
    else:
        global_start, global_stop = 0, ds_size

    interval = (global_stop - global_start)
    if interval % world_size == 0:
        chunk_size = int(interval // world_size)
    else:
        chunk_size = int(interval // world_size) + 1

    local_start = global_start + rank_index * chunk_size
    local_stop = min(local_start + chunk_size, global_stop)

    return local_start, local_stop

def main():
    args = parse_args()

    rank_index = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f"cuda:{rank_index}")

    fs_utils.create_dir_if_not_exists(args.out_dir)
    test_transform = get_test_transform(args.dataset)
    model = torch.load(args.normal_aug_model_path)
    model = model.to(device)

    if args.dataset == 'cifar100':
        dataset = CIFAR100(root=base_settings.DATA_ROOT, download=True, train=True, transform=test_transform)
    else:
        dataset = MiniImageNet(train=True, transform=test_transform, reduce_dim=False)

    global_index_range = [int(r) for r in args.global_index_range.split(':')] if args.global_index_range else None
    index_range = prepare_index_range(len(dataset), rank_index, world_size, global_index_range)
    print(f"Computed indices: ({index_range[0]}, {index_range[1]}) for rank-index: {rank_index}")

    result = run_certification(dataset, index_range, model, args.sigma, args.batch_size, device)
    np.save(op.join(args.out_dir, f"result_{index_range[0]}_{index_range[1]}.npy"), result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['cifar100', 'mini_imagenet'])
    parser.add_argument('normal_aug_model_path')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('out_dir')
    parser.add_argument('sigma', type=float)
    parser.add_argument('--global_index_range')
    return parser.parse_args()


if __name__ == '__main__':
    main()
