import argparse

import numpy as np

from utils.logger import log
from utils import fs_utils


@log
def prepare_splits_for_bins(densities_arr, noisy_densities_arr, bin_size):
    if noisy_densities_arr is not None:
        diffs = np.expand_dims(densities_arr, axis=1) - noisy_densities_arr
        quantity_to_sort = np.mean(diffs, axis=1)
    else:
        quantity_to_sort = densities_arr

    examples_sorted = np.argsort(quantity_to_sort)
    splits_dict = {'train': [], 'val': []}
    for bin_start_index in range(0, len(examples_sorted), bin_size):
        val_indices = list(examples_sorted[bin_start_index: (bin_start_index + bin_size)])
        train_indices = list(examples_sorted[:bin_start_index]) + list(examples_sorted[(bin_start_index + bin_size):])

        splits_dict['train'].append(train_indices)
        splits_dict['val'].append(val_indices)

    prepare_splits_for_bins.logger.info("Created %d splits" % len(splits_dict['train']))
    return splits_dict


def main():
    args = parse_args()

    densities_arr = np.array(fs_utils.read_json(args.densities_path), dtype=np.float32)
    if args.noise_densities_path:
        noisy_densities_arr = np.load(args.noise_densities_path)
    else:
        noisy_densities_arr = None

    splits_dict = prepare_splits_for_bins(densities_arr, noisy_densities_arr, args.bin_size)
    np.savez(args.out_splits_npz_path, **{'train': np.array(splits_dict['train'], dtype=np.int32),
                                          'val': np.array(splits_dict['val'], dtype=np.int32)})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_splits_npz_path')
    parser.add_argument('densities_path')
    parser.add_argument('--noise_densities_path')
    parser.add_argument('--bin_size', type=int, default=1000)
    return parser.parse_args()


if __name__ == '__main__':
    main()
