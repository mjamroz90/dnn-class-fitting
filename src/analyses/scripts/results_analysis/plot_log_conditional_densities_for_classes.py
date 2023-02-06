import argparse
import os
import os.path as op

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import fs_utils
from utils import logger
from src.analyses.scripts.results_analysis import res_analysis_utils as ra_utils

# USED IN AAAI PAPER

X_LABEL = 'Log-density'
Y_LABEL = 'Classes'


@logger.log
def generate_stats_for_diffs(diffs, ex_labels):
    labels_num = len(set(ex_labels))
    classes_diffs = [[] for _ in range(labels_num)]

    for i in range(diffs.shape[0]):
        i_label = int(ex_labels[i])
        i_diffs = [float(f) for f in diffs[i, :]]
        classes_diffs[i_label].extend(i_diffs)

    classes_means, classes_stds = [], []
    for class_, class_diffs in enumerate(classes_diffs):
        class_mean = np.mean(class_diffs)
        class_std = np.std(class_diffs)

        classes_means.append(float(class_mean))
        classes_stds.append(float(class_std))

    generate_stats_for_diffs.logger.info("Collected means and stds for all %d classes" % labels_num)
    return classes_means, classes_stds


@logger.log
def generate_stats_for_densities(densities, ex_labels):
    labels_num = len(set(ex_labels))
    classes_densities = [[] for _ in range(labels_num)]

    for i in range(densities.shape[0]):
        i_label = int(ex_labels[i])
        classes_densities[i_label].append(densities[i])

    classes_means, classes_stds = [], []
    for class_, class_densities in enumerate(classes_densities):
        class_mean = np.mean(class_densities)
        class_std = np.std(class_densities)

        classes_means.append(float(class_mean))
        classes_stds.append(float(class_std))

    generate_stats_for_densities.logger.info("Collected means and stds for all %d classes" % labels_num)
    return classes_means, classes_stds


@logger.log
def prepare_plot(densities, ex_labels, out_plot_file, title, draw_line_with_text=False, dataset=None):
    sns.set_theme(style='darkgrid')
    sns.set_context('paper')

    plt.xlabel(X_LABEL, fontsize=24)
    plt.ylabel(Y_LABEL, fontsize=24)
    plt.title(title, fontsize=24)

    classes_means, classes_stds = generate_stats_for_densities(densities, ex_labels)
    classes_means_stds = zip(classes_means, classes_stds)
    classes_means_stds_sorted = sorted(classes_means_stds, key=lambda x: x[0])

    classes_means_sorted, classes_stds_sorted = zip(*classes_means_stds_sorted)

    plt.errorbar(x=list(classes_means_sorted), y=list(range(len(classes_means))), xerr=list(classes_stds_sorted),
                 ls='none', color='b', marker='o', markersize=2, elinewidth=0.6)

    if draw_line_with_text:
        dataset_text_coords = {ra_utils.CIFAR100_NAME: [(-22., 40.), (-75., 70.)],
                               ra_utils.MINI_IMAGENET_NAME: [(-30., 30.), (-80., 70.)]}
        axis = plt.gca()
        plt.axvline(x=__find_separating_line_coord(classes_means_sorted), color='r', linewidth=2, linestyle="--")
        hd_coords, ld_coords = dataset_text_coords[dataset][0], dataset_text_coords[dataset][1]
        axis.text(*hd_coords, "High-density\n classes", fontdict={'size': 18})
        axis.text(*ld_coords, "Low-density\n classes", fontdict={'size': 18})

    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)

    plt.tight_layout()
    plt.savefig(out_plot_file)

    plt.clf()


def __find_separating_line_coord(classes_means):
    interval_lengths_with_index = [(i, classes_means[i+1] - classes_means[i]) for i in range(len(classes_means) - 1)]
    biggest_interval = max(interval_lengths_with_index, key=lambda x: x[1])
    biggest_i = biggest_interval[0]
    half_length = (classes_means[biggest_i + 1] - classes_means[biggest_i]) / 2.
    return classes_means[biggest_i] + half_length


def main():
    args = parse_args()
    ex_labels = np.load(args.mem_scores_npz_path)['tr_labels']

    fs_utils.create_dir_if_not_exists(args.out_plots_dir)
    densities_file_names = [p for p in os.listdir(args.densities_dir) if p.endswith(('json',))]

    dataset = ra_utils.find_dataset_name(args.densities_dir)
    for densities_file_name in densities_file_names:
        layer_index = int(densities_file_name.split('_')[2])

        densities_path = op.join(args.densities_dir, densities_file_name)

        densities = np.array(fs_utils.read_json(densities_path), dtype=np.float32)
        out_plot_file = op.join(args.out_plots_dir, "%s.%s" % (op.splitext(densities_file_name)[0], args.plot_ext))

        draw_line_with_text = layer_index > 2 and not args.no_additional_info_last_layer
        title = ra_utils.layer_ds_title(dataset, layer_index, args.no_additional_info_last_layer)

        prepare_plot(densities, ex_labels, out_plot_file, title,
                     draw_line_with_text, dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('densities_dir')
    parser.add_argument('mem_scores_npz_path')
    parser.add_argument('out_plots_dir')
    parser.add_argument('--plot_ext', choices=['png', 'eps', 'pdf'], required=True)
    parser.add_argument('--no_additional_info_last_layer', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
