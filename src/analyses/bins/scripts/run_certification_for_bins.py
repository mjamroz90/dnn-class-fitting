import os
import os.path as op
import argparse
from collections import OrderedDict

import torch
import numpy as np

from src.analyses.bins.scripts.run_certification import (
    run_certification,
    prepare_index_range,
    get_dataset
)
from src.transforms import get_test_transform
from utils import fs_utils


def calculate_certification_for_splits(splits_val, models_mapping, dataset, device, sigma, batch_size):
    print(f"Processing splits chunks of shape: {splits_val.shape}, with models mapping of size: {len(models_mapping)}")
    test_transform = get_test_transform(dataset, True)
    result = []
    for split_index, split_model_path in models_mapping.items():
        split_val_indices = splits_val[int(split_index), :]

        model = torch.load(split_model_path, map_location=device)
        split_val_dataset = get_dataset(dataset, test_transform, False, split_val_indices)
        split_certification = run_certification(split_val_dataset, (0, len(split_val_indices)), model, sigma,
                                                batch_size, device)
        result.append(split_certification)

    return np.array(result, dtype=np.float32)


def choose_models_subset(models_mapping, index_range):
    start, stop = index_range
    models_mapping_subset = OrderedDict({str(split_index): models_mapping[str(split_index)]
                             for split_index in range(start, stop, 1)})
    return models_mapping_subset


def main():
    args = parse_args()
    splits_npz_path = op.join(args.certification_root_dir, 'splits.npz')
    models_mapping_path = op.join(args.certification_root_dir, 'models_mapping.json')

    fs_utils.create_dir_if_not_exists(op.join(args.certification_root_dir, args.out_dir_name))

    rank_index = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f"cuda:{rank_index}")

    splits_val = np.load(splits_npz_path)['val']
    models_mapping = fs_utils.read_json(models_mapping_path)

    global_index_range = [int(r) for r in args.global_index_range.split(':')] if args.global_index_range else None
    index_range = prepare_index_range(int(splits_val.shape[0]), rank_index, world_size, global_index_range)
    print(f"Computed range indices: ({index_range[0]}, {index_range[1]}) for rank-index: {rank_index}")

    models_mapping_subset = choose_models_subset(models_mapping, index_range)
    result = calculate_certification_for_splits(splits_val, models_mapping_subset, args.dataset,
                                                device, args.sigma, args.batch_size)
    np.save(op.join(args.certification_root_dir, args.out_dir_name,
                    f"result_{index_range[0]}_{index_range[1]}.npy"), result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['cifar100', 'mini_imagenet'])
    parser.add_argument('certification_root_dir')
    parser.add_argument('out_dir_name')
    parser.add_argument('sigma', type=float)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('--global_index_range')
    return parser.parse_args()


if __name__ == '__main__':
    main()
