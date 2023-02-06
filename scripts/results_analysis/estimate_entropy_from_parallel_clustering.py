import argparse
import os.path as op

import numpy as np
from tqdm import tqdm

from scripts.results_analysis.estimate_entropy_from_clustering import choose_trace_files
from scripts.results_analysis.calculate_predictive_log_densities import split_parallel_trace_per_clustering
from src.models.clustering import entropy_estimation
from utils import fs_utils
from utils import logger


@logger.log
def estimate_entropy_for_classes(trace_paths, class_indices_split, samples_num):
    data_trace_path = op.join(op.dirname(trace_paths[0]), 'cgs_0.pkl')
    data_arr = fs_utils.read_pickle(data_trace_path)['data']

    traces_classes_entropy_vals = []
    for trace_path in trace_paths:
        estimate_entropy_for_classes.logger.info("Started to process trace path: %s" % trace_path)

        parallel_trace_obj = fs_utils.read_pickle(trace_path)
        classes_trace_objs = split_parallel_trace_per_clustering(parallel_trace_obj)
        classes_entropy_vals = []
        for class_index, class_split in tqdm(enumerate(class_indices_split)):
            class_data_arr = data_arr[class_index]
            class_trace_obj = classes_trace_objs[class_index]

            entropy_obj = entropy_estimation.EntropyEstimator(class_trace_obj, samples_num, 'relative',
                                                              **{'data': class_data_arr})
            class_distr_entropy = float(entropy_obj.estimate_entropy_with_sampling())
            classes_entropy_vals.append(class_distr_entropy)

        traces_classes_entropy_vals.append(classes_entropy_vals)

    traces_classes_entropy_vals_arr = np.array(traces_classes_entropy_vals, dtype=np.float32)
    return [float(v) for v in np.mean(traces_classes_entropy_vals_arr, axis=0)]


def main():
    args = parse_args()
    trace_paths = choose_trace_files(args.clustering_dir, args.start_iteration, args.interval)
    class_indices_split = fs_utils.read_json(args.class_indices_split_path)
    classes_entropies = estimate_entropy_for_classes(trace_paths, class_indices_split, args.samples_num)

    fs_utils.write_json(classes_entropies, args.out_json_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('clustering_dir')
    parser.add_argument('start_iteration', type=int)
    parser.add_argument('interval', type=int)
    parser.add_argument('class_indices_split_path')
    parser.add_argument('out_json_path')
    parser.add_argument('--samples_num', type=int, default=10 ** 5)
    return parser.parse_args()


if __name__ == '__main__':
    main()
