import argparse
import os
import os.path as op

import numpy as np
from sklearn import metrics

from scripts.results_analysis import estimate_entropy_from_clustering
from scripts.results_analysis import calculate_predictive_log_densities
from utils import fs_utils
from utils import logger


def calculate_densities(activations, trace_obj):
    individual_traces = calculate_predictive_log_densities.split_parallel_trace_per_clustering(trace_obj)
    acts_class_densities = []
    for class_trace in individual_traces:
        calculator_obj = calculate_predictive_log_densities.DensityCalculator([class_trace])
        densities = calculator_obj.calculate_log_densities(activations)
        acts_class_densities.append(densities)

    return np.array(acts_class_densities, dtype=np.float32)


@logger.log
def check_class_stats_for_indices(activations_arr, indices, trace_obj, gts):
    indices_activations = activations_arr[indices, :]
    acts_class_densities = calculate_densities(indices_activations, trace_obj)
    check_class_stats_for_indices.logger.info("Calculated densities for indices of shape: %s" %
                                              str(acts_class_densities.shape))
    indices_preds = np.argmax(acts_class_densities, axis=0).astype(np.int32)
    acc = metrics.accuracy_score(gts, indices_preds)
    prec = metrics.precision_score(gts, indices_preds, average='weighted')
    rec = metrics.precision_score(gts, indices_preds, average='weighted')
    f1 = metrics.f1_score(gts, indices_preds, average='weighted')

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}


def class_stats(examples_classes):
    classes_counts = {}
    for class_ in examples_classes:
        if class_ not in classes_counts:
            classes_counts[class_] = 1
        else:
            classes_counts[class_] += 1

    classes_counts_sorted = dict(sorted(list(classes_counts.items()), key=lambda x: x[1]))
    return classes_counts_sorted


def do_classification_experiment_with_existing_examples(activations_path, splits_dir, trace_obj, examples_classes):
    activations_arr = np.load(activations_path)
    least_memorized_path = op.join(splits_dir, 'least_memorized.json')
    most_memorized_path = op.join(splits_dir, 'most_memorized.json')
    random_path = op.join(splits_dir, 'random.json')

    least_mem_indices = fs_utils.read_json(least_memorized_path)
    most_mem_indices = fs_utils.read_json(most_memorized_path)
    random_indices = fs_utils.read_json(random_path)

    most_memorized_stats = check_class_stats_for_indices(activations_arr, most_mem_indices, trace_obj,
                                                         examples_classes[most_mem_indices])
    least_memorized_stats = check_class_stats_for_indices(activations_arr, least_mem_indices, trace_obj,
                                                          examples_classes[least_mem_indices])
    random_stats = check_class_stats_for_indices(activations_arr, random_indices, trace_obj,
                                                 examples_classes[random_indices])

    return most_memorized_stats, least_memorized_stats, random_stats


def avg_results(results_list):
    return {'acc': np.mean([r['acc'] for r in results_list]), 'prec': np.mean([r['prec'] for r in results_list]),
            'rec': np.mean([r['rec'] for r in results_list]), 'f1': np.mean([r['f1'] for r in results_list])}

@logger.log
def main():
    args = parse_args()
    acts_arr_paths = [op.join(args.activations_dir, p) for p in os.listdir(args.activations_dir)
                      if p.endswith(('ld.npy',))]
    traces_dirs = [op.join(args.clustering_traces_dir, op.splitext(op.basename(p))[0]) for p in acts_arr_paths]
    ref_matrix = np.load(args.ref_memorization_matrix_path)
    result = {}

    for acts_arr_path, trace_dir in zip(acts_arr_paths, traces_dirs):
        trace_paths = estimate_entropy_from_clustering.choose_trace_files(trace_dir, args.init_iteration, args.interval)
        most_mem_trace_results, least_mem_trace_results, random_trace_results = [], [], []
        for trace_path in trace_paths:
            trace_obj = fs_utils.read_pickle(trace_path)
            most_mem_stats, least_mem_stats, random_stats = do_classification_experiment_with_existing_examples(
                    acts_arr_path, args.splits_dir, trace_obj, ref_matrix['tr_labels'])
            most_mem_trace_results.append(most_mem_stats)
            least_mem_trace_results.append(least_mem_stats)
            random_trace_results.append(random_stats)

        avg_most_mem_stats = avg_results(most_mem_trace_results)
        avg_least_mem_stats = avg_results(least_mem_trace_results)
        avg_random_stats = avg_results(random_trace_results)

        main.logger.info("CLASSIFICATION REPORT FOR ACTIVATIONS: %s" % op.basename(acts_arr_path))
        main.logger.info("Most memorized stats: %s" % str(avg_most_mem_stats))
        main.logger.info("Least memorized stats: %s" % str(avg_least_mem_stats))
        main.logger.info("Random stats: %s" % str(avg_random_stats))

        result[op.splitext(op.basename(acts_arr_path))[0]] = {'most_memorized': most_mem_trace_results,
                                                              'least_memorized': least_mem_trace_results,
                                                              'random': random_trace_results}

    fs_utils.write_json(result, args.out_results_json_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('activations_dir')
    parser.add_argument('splits_dir')
    parser.add_argument('clustering_traces_dir')
    parser.add_argument('init_iteration', type=int)
    parser.add_argument('interval', type=int)
    parser.add_argument('ref_memorization_matrix_path')
    parser.add_argument('out_results_json_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()
