import argparse

import numpy as np
import tensorflow as tf

from scripts.results_analysis import calculate_predictive_log_densities
from scripts.results_analysis import estimate_entropy_from_clustering
from src.models.clustering import latent_space_sampler
from utils import fs_utils
from utils import logger


@logger.log
def calculate_dkls_for_classes(classes_traces):
    log_probs = np.zeros((len(classes_traces), len(classes_traces)), dtype=np.float32)
    dkls = np.zeros((len(classes_traces), len(classes_traces)), dtype=np.float32)

    densities = [latent_space_sampler.LatentSpaceSampler(trace).prepare_t_student_mixture() for trace in classes_traces]
    classes_samples = [density.sample(sample_shape=10**4,) for density in densities]
    for class_index, class_density in enumerate(densities):
        # class_log_probs.shape -> (classes_num)
        class_log_probs = tf.reduce_mean(class_density.log_prob(classes_samples), axis=1).numpy()
        # log_probs[i, j] = log_probabilities evaluated by class-i density on class-j samples
        log_probs[class_index, :] = class_log_probs

    for i in range(len(classes_traces)):
        for j in range(len(densities)):
            if i == j:
                continue

            ij_dkl = float(log_probs[i, i] - log_probs[j, i])
            dkls[i, j] = ij_dkl

    calculate_dkls_for_classes.logger.info("Calculated dkls for all classes combinations, for classes length: %d" %
                                           len(classes_traces))
    return dkls


@logger.log
def calculate_dkls_between_classes(traces_paths):
    traces_objs = [fs_utils.read_pickle(tp) for tp in traces_paths]
    dkls = []
    for trace_index, trace_obj in enumerate(traces_objs):
        classes_traces = calculate_predictive_log_densities.split_parallel_trace_per_clustering(trace_obj)
        dkls.append(calculate_dkls_for_classes(classes_traces))

        calculate_dkls_between_classes.logger.info("Calculated dkls for trace index: %d" % trace_index)

    return np.mean(dkls, axis=0)


def main():
    args = parse_args()

    traces_paths = estimate_entropy_from_clustering.choose_trace_files(args.clustering_results_dir,
                                                                       args.init_iteration, args.interval)
    dkls = calculate_dkls_between_classes(traces_paths)
    np.save(args.out_arr_path, dkls)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('clustering_results_dir')
    parser.add_argument('init_iteration', type=int)
    parser.add_argument('interval', type=int)
    parser.add_argument('out_arr_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()
