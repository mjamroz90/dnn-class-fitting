import argparse
import os
import os.path as op

import numpy as np
import pandas as pd

from utils import logger
from src.analyses.scripts.results_analysis import res_analysis_utils as ra_utils

Y_LABEL = 'KL divergence'
X_LABEL = 'Stage number'


@logger.log
def process_type_dir(type_dir, acts_type):
    dkls_npy_files = [op.join(type_dir, p) for p in os.listdir(type_dir) if p.startswith((acts_type,))]
    dkls_npy_files = sorted(dkls_npy_files, key=lambda x: int(op.basename(x).split('_')[2]))
    results = []

    for dkls_npy_file in dkls_npy_files:
        dkls_arr = np.load(dkls_npy_file)

        rows, cols = np.where(dkls_arr > 0.)
        results.append(list(dkls_arr[rows, cols]))

        process_type_dir.logger.info("Processed %s dkls npy array" % dkls_npy_file)

    return results


def generate_pd_data_frame(result_dict, series_excluded):
    type_dir_to_series_mapping = {'dkls_least_mem': 'least memorized', 'dkls_most_mem': 'most memorized',
                                  'dkls_random': 'random'}
    layer_index_title, layer_dkl_title, examples_title = X_LABEL, Y_LABEL, 'Examples'
    data = {layer_index_title: [], layer_dkl_title: [], examples_title: []}

    for type_dir, type_dir_results in result_dict.items():
        if type_dir_to_series_mapping[type_dir] in series_excluded:
            continue

        for layer_index, layer_dkls in enumerate(type_dir_results):
            data[layer_index_title].extend([layer_index + 1] * len(layer_dkls))
            data[layer_dkl_title].extend(layer_dkls)
            data[examples_title].extend([type_dir_to_series_mapping[type_dir]] * len(layer_dkls))

    return pd.DataFrame(data=data), {'x': layer_index_title, 'y': layer_dkl_title,
                                     'hue': examples_title, 'palette': ra_utils.EXAMPLES_PALETTE,
                                     'marker': 'o'}


def plot_with_seaborn(data_frame, plot_args, out_plot_file, title):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style='darkgrid')
    sns.set_context('paper')
    plt.tight_layout()

    line_plot = sns.lineplot(data=data_frame, **plot_args, err_style='band', ci='sd')
    line_plot.get_legend().set_title(None)
    line_plot.set_title(title, fontsize=ra_utils.TITLE_FONT_SIZE)
    line_plot.set_xlabel(X_LABEL, fontsize=ra_utils.AXES_FONT_SIZE)
    line_plot.set_ylabel(Y_LABEL, fontsize=ra_utils.AXES_FONT_SIZE)
    line_plot.set_xticks([1, 2, 3, 4])
    line_plot.figure.savefig(out_plot_file)


def main():
    args = parse_args()
    types_dirs = [op.join(args.dkls_results_root_dir, d) for d in os.listdir(args.dkls_results_root_dir)
                  if d.startswith('dkls')]

    results_dict = {}
    for type_dir in types_dirs:
        type_dir_results = process_type_dir(type_dir, args.acts_type)
        results_dict[op.basename(type_dir)] = type_dir_results

    plot_df, plot_args = generate_pd_data_frame(results_dict, args.exclude_series)
    plot_with_seaborn(plot_df, plot_args, args.out_plots_file, ra_utils.find_dataset_name(args.out_plots_file))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dkls_results_root_dir')
    parser.add_argument('out_plots_file')
    parser.add_argument('--acts_type', choices=['max', 'avg'])
    parser.add_argument('--exclude_series', nargs='+')
    return parser.parse_args()


if __name__ == '__main__':
    main()
