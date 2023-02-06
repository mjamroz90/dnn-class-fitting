import argparse
import os.path as op
import os

import numpy as np

from scripts.results_analysis import estimate_entropy_from_clustering
from src.models.clustering import entropy_estimation
from utils import fs_utils
from utils import logger


@logger.log
class DensityCalculator:

    def __init__(self, trace_objs):
        self.trace_objs = trace_objs
        self.traces_estimators = [entropy_estimation.EntropyEstimator(to, samples_num=None, entropy_type='differential')
                                  for to in self.trace_objs]
        self.logger.info("Loaded %d traces from paths" % len(self.traces_estimators))

    def calculate_log_densities(self, activations_arr):
        traces_log_densities = []
        for i, trace_estimator in enumerate(self.traces_estimators):
            trace_mixture_distr = trace_estimator.t_student_mixture
            log_densities_for_activations = trace_mixture_distr.log_prob(activations_arr).numpy()

            self.logger.info("Calculated log-densities for %d activation vectors with %d trace" %
                             (log_densities_for_activations.shape[0], i))
            traces_log_densities.append(log_densities_for_activations)

        traces_log_densities_arr = np.array(traces_log_densities, dtype=np.float32)
        return np.mean(traces_log_densities_arr, axis=0)


@logger.log
def calculate_log_densities(activations_path, traces_objs):
    activations_arr = np.load(activations_path)
    density_calculator = DensityCalculator(traces_objs)

    shape = activations_arr.shape
    if len(shape) == 3:
        n, s, d = shape
        log_densities = np.zeros((n, s), dtype=np.float32)
        for idx in range(s):
            sample = np.squeeze(activations_arr[:, idx, :])
            log_densities[:, idx] = density_calculator.calculate_log_densities(sample)
            calculate_log_densities.logger.info("Calculated log-densities for sample: %d" % idx)
    else:
        log_densities = density_calculator.calculate_log_densities(activations_arr)

    return log_densities


@logger.log
def calculate_log_densities_for_parallel(activations_paths, parallel_traces_objs, class_indices_split,
                                         calculate_all_classes):
    activations_arr = np.load(activations_paths)
    parallel_traces_objs = [split_parallel_trace_per_clustering(to) for to in parallel_traces_objs]
    classes_num = len(class_indices_split)

    if len(activations_arr.shape) == 3:
        n, s, d = activations_arr.shape
        examples_densities = np.zeros((n, s), dtype=np.float32)
        examples_densities_all_classes = np.zeros((n, s, classes_num), dtype=np.float32)
        cubic_shape = True
    else:
        examples_densities = np.zeros((activations_arr.shape[0],), dtype=np.float32)
        examples_densities_all_classes = np.zeros((activations_arr.shape[0], classes_num), dtype=np.float32)
        cubic_shape = False

    for class_index, class_examples in enumerate(class_indices_split):
        class_traces_objs = [pto[class_index] for pto in parallel_traces_objs]
        class_density_calculator = DensityCalculator(class_traces_objs)

        if cubic_shape:
            class_activations_arr = activations_arr[class_examples, :, :]
            examples_densities[class_examples, :] = class_density_calculator.calculate_log_densities(
                class_activations_arr)
            if calculate_all_classes:
                examples_densities_all_classes[:, :, class_index] = class_density_calculator.calculate_log_densities(
                        activations_arr)
        else:
            class_activations_arr = activations_arr[class_examples, :]
            examples_densities[class_examples] = class_density_calculator.calculate_log_densities(class_activations_arr)
            if calculate_all_classes:
                examples_densities_all_classes[:, class_index] = class_density_calculator.calculate_log_densities(
                    activations_arr)

        calculate_log_densities_for_parallel.logger.info("Calculated log-densities for class: %d" % class_index)

    return examples_densities, examples_densities_all_classes


def split_parallel_trace_per_clustering(trace_obj):
    clustering_nums = len(trace_obj['cluster_params'])
    individual_clustering_traces = []
    for i in range(clustering_nums):
        i_clustering_trace = {'cluster_params': trace_obj['cluster_params'][i],
                              'alpha': trace_obj['alpha'][i],
                              'cluster_assignment': trace_obj['cluster_assignment'][i]}
        individual_clustering_traces.append(i_clustering_trace)

    return individual_clustering_traces


def save_densities(densities, out_path, is_npy):
    if is_npy:
        np.save(out_path, densities)
    else:
        densities_list = [float(f) for f in densities]
        fs_utils.write_json(densities_list, out_path)


def main():
    args = parse_args()

    if args.no_all_classes_calc and not args.class_indices_split_path:
        raise ValueError("Option --no_all_classes_calc can be passed only in per-class mode")

    input_path_suffix = args.input_path_suffix if args.input_path_suffix else ""
    input_path_prefix = args.input_path_prefix if args.input_path_prefix else ""
    layers_indices = [int(i) for i in args.layers_indices.split(',')] if args.layers_indices else None

    if op.isdir(args.activations_path_or_dir):
        activations_paths = [op.join(args.activations_path_or_dir, p) for p in
                             os.listdir(args.activations_path_or_dir) if p.endswith(("%s.npy" % input_path_suffix,))
                             and p.startswith((input_path_prefix,))]
        fs_utils.create_dir_if_not_exists(args.out_results_file)
    else:
        activations_paths = [args.activations_path_or_dir]

    for act_path in activations_paths:
        if layers_indices is not None:
            act_index = int(op.basename(act_path).split('_')[2])
            if act_index not in layers_indices:
                continue

        if not op.isdir(args.out_results_file):
            out_act_path, out_file_name = args.out_results_file, op.basename(args.out_results_file)
            clustering_act_dir = args.clustering_dir
        else:
            out_file_name = '_'.join(op.splitext(op.basename(act_path))[0].split('_')[:3])
            if args.out_results_suffix:
                out_file_name += "_%s" % args.out_results_suffix

            out_file_name += (".npy" if args.npy else ".json")

            out_act_path = op.join(args.out_results_file, out_file_name)
            clustering_act_dir = op.join(args.clustering_dir, op.splitext(op.basename(act_path))[0])

        traces_paths = estimate_entropy_from_clustering.choose_trace_files(clustering_act_dir, args.init_iteration,
                                                                           args.interval)
        traces_objs = [fs_utils.read_pickle(tp) for tp in traces_paths]

        if args.class_indices_split_path:
            class_indices_split = fs_utils.read_json(args.class_indices_split_path)
            log_densities, log_densities_all_classes = calculate_log_densities_for_parallel(act_path, traces_objs,
                                                                                            class_indices_split,
                                                                                        not args.no_all_classes_calc)
            save_densities(log_densities, out_act_path, args.npy)
            if not args.no_all_classes_calc:
                out_all_classes_file = op.join(op.dirname(out_act_path),
                                               "%s_all_classes.npy" % op.splitext(out_file_name)[0])
                save_densities(log_densities_all_classes, out_all_classes_file, True)
        else:
            log_densities = calculate_log_densities(act_path, traces_objs)
            save_densities(log_densities, out_act_path, args.npy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('activations_path_or_dir')
    parser.add_argument('clustering_dir')
    parser.add_argument('init_iteration', type=int)
    parser.add_argument('interval', type=int)
    parser.add_argument('out_results_file')
    parser.add_argument('--out_results_suffix')
    parser.add_argument('--class_indices_split_path')
    parser.add_argument('--input_path_suffix')
    parser.add_argument('--input_path_prefix')
    parser.add_argument('--npy', action='store_true')
    parser.add_argument('--layers_indices')
    parser.add_argument('--no_all_classes_calc', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
