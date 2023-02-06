import os
import os.path as op

from src.models.clustering import entropy_estimation
from utils import logger


@logger.log
def choose_trace_files(clustering_dir, init_iteration, interval):
    trace_files = [p for p in os.listdir(clustering_dir) if p.endswith(('pkl',))]

    init_trace_file = op.join(clustering_dir, "cgs_%d.pkl" % init_iteration)
    trace_files_num = len(trace_files)
    choose_trace_files.logger.info("Listed %d trace pkl files from directory: %s" % (trace_files_num, clustering_dir))

    max_iter = max([int(op.splitext(x)[0].split('_')[-1]) for x in trace_files])
    if not op.exists(init_trace_file):
        choose_trace_files.logger.warn("File %s (initial iteration) does not exist" % init_trace_file)

    chosen_trace_files = []
    if init_iteration >= max_iter:
        curr_iteration, interval = max(max_iter - 50, 1), 1
        max_iter -= 2
        choose_trace_files.logger.info("Init-iteration %d smaller than max-iteration %d, "
                                       "new init-iteration: %d and interval: %d" % (init_iteration, max_iter,
                                                                                    curr_iteration, interval))
    else:
        curr_iteration = init_iteration

    while curr_iteration <= max_iter:
        curr_trace_file = op.join(clustering_dir, "cgs_%d.pkl" % curr_iteration)
        if op.exists(curr_trace_file):
            chosen_trace_files.append(curr_trace_file)

        curr_iteration += interval

    choose_trace_files.logger.info("Chosen trace files: %s" % str(chosen_trace_files))
    choose_trace_files.logger.info("Chosen trace files num: %d" % len(chosen_trace_files))

    return chosen_trace_files


@logger.log
def do_entropy_estimation_for_traces(trace_paths, samples_num, entropy_type):
    entropy_results = []
    for trace_path in trace_paths:
        do_entropy_estimation_for_traces.logger.info("Started entropy estimation for path: %s" % trace_path)

        if entropy_type == 'relative':
            data_trace_path = op.join(op.dirname(trace_path), 'cgs_0.pkl')
            kwargs = {'data_trace_path': data_trace_path}
        else:
            kwargs = {}

        estimator = entropy_estimation.EntropyEstimator(trace_path, samples_num, entropy_type, **kwargs)
        entropy_val = estimator.estimate_entropy_with_sampling()

        entropy_results.append(float(entropy_val))

    do_entropy_estimation_for_traces.logger.info("Computed all of the entropy values, mean: %.3f"
                                                 % (sum(entropy_results) / len(entropy_results)))
    return entropy_results


def collect_entropy_val_info(net_clustering_dir, start_iteration, interval, samples_num, entropy_type):
    layers_dirs = [d for d in os.listdir(net_clustering_dir) if op.isdir(op.join(net_clustering_dir, d))]
    layers_dirs_sorted = sorted(layers_dirs, key=lambda x: int(x.split('_')[2]))
    layers_dirs_sorted = [op.join(net_clustering_dir, d) for d in layers_dirs_sorted]

    layers_dirs_entropy_vals = []
    for layer_dir in layers_dirs_sorted:
        chosen_trace_paths = choose_trace_files(layer_dir, start_iteration, interval)
        entropy_results = do_entropy_estimation_for_traces(chosen_trace_paths, samples_num, entropy_type)
        layers_dirs_entropy_vals.append(entropy_results)

    return layers_dirs_entropy_vals

