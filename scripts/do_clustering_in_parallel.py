import argparse
import os.path as op

import numpy as np

from src.models.clustering import parallel_collapsed_gibbs_sampler
from utils import fs_utils
from utils.logger import log


@log
def main():
    args = parse_args()
    data = np.load(args.in_npy_arr_file)

    main.logger.info("Loaded data from %s, shape: %s" % (op.abspath(args.in_npy_arr_file), str(data.shape)))
    fs_utils.create_dir_if_not_exists(args.out_dir)

    class_indices_split = fs_utils.read_json(args.class_indices_split_path)
    main.logger.info("Loaded class indices split info, class counts: %s" % str([len(c) for c in class_indices_split]))

    parallel_cgs = parallel_collapsed_gibbs_sampler.ParallelCollapsedGibbsSampler(init_strategy='init_data_stats',
                            max_clusters_num=args.init_clusters_num, out_dir=args.out_dir,  batch_size=args.batch_size,
                            **{'skip_epochs_logging': args.skip_epochs_logging,
                                'skip_epochs_ll_calc': args.skip_epochs_ll_calc,
                                'restore_snapshot_pkl_path': args.snapshot_path if args.snapshot_path else None})

    data_chunks = [data[class_indices, :] for class_indices in class_indices_split]
    parallel_cgs.fit(args.iterations_num, data_chunks)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_npy_arr_file')
    parser.add_argument('class_indices_split_path')
    parser.add_argument('out_dir')
    parser.add_argument('iterations_num', type=int)
    parser.add_argument('init_clusters_num', type=int)
    parser.add_argument('--skip_epochs_logging', type=int, default=40)
    parser.add_argument('--skip_epochs_ll_calc', type=int, default=40)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--snapshot_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()
