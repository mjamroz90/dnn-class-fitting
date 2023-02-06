import argparse
import os.path as op

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.analyses.scripts.results_analysis import res_analysis_utils as ra_utils
from utils import fs_utils
from src.analyses.scripts.results_analysis.plot_log_conditional_densities_for_classes import (
    generate_stats_for_densities
)

# USED IN AAAI PAPER

X_LABEL = "Mean log-density"
Y1_LABEL = "Mean distance"
Y2_LABEL = "Relative entropy"


def find_mean(class_acts_arr):
    distances = euclidean_distances(class_acts_arr, class_acts_arr)
    distances_non_zero = distances[distances > 0.]
    mean_val = np.mean(distances_non_zero)
    return mean_val


def prepare_acts_distances_data(acts_arr, ex_labels):
    mean_distances = []
    for class_ in range(100):
        class_examples_mask = (ex_labels == class_)
        class_acts_arr = acts_arr[class_examples_mask, :]

        class_mean = find_mean(class_acts_arr)
        mean_distances.append(float(class_mean))

    return mean_distances


def plot_density_dist_relation(densities_arr, acts_arr, acts_entropies, ex_labels, out_plot_path, title):
    class_densities_means, _ = generate_stats_for_densities(densities_arr, ex_labels)
    class_mean_distances = prepare_acts_distances_data(acts_arr, ex_labels)

    sns.set_theme(style='darkgrid')
    sns.set_context('paper')
    sns.set(rc={"figure.figsize": (9., 4.)})

    red_line_style = {"marker": "x", "markersize": 3, "linestyle": 'None'}
    blue_line_style = {"marker": "o", "markersize": 3, "linestyle": 'None'}

    ax = plt.gca()
    ax.set_xlabel(X_LABEL, fontsize=21)
    ax.set_ylabel(Y1_LABEL, fontsize=21)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    first_plot = ax.plot(class_densities_means, class_mean_distances, color='b', label=Y1_LABEL, **blue_line_style)

    secondary_ax = ax.twinx()
    secondary_ax.set_ylabel(Y2_LABEL, fontsize=21)
    secondary_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    second_plot = secondary_ax.plot(class_densities_means, acts_entropies, color='r', label=Y2_LABEL, **red_line_style)

    ax.set_yticks(np.linspace(ax.get_ybound()[0], ax.get_ybound()[1], 6))
    plot_regression_line(class_densities_means, acts_entropies, secondary_ax)

    secondary_ax.set_yticks(np.linspace(secondary_ax.get_ybound()[0], secondary_ax.get_ybound()[1], 6))

    plt.title(title, fontsize=21)
    ax.legend(first_plot + second_plot, [l_.get_label() for l_ in first_plot + second_plot],
              loc="upper left", prop={'size': 12})

    plt.tight_layout()
    plt.savefig(out_plot_path)
    plt.clf()


def plot_regression_line(x, y, axis):
    a, b = np.polyfit(x, y, deg=1)
    x_seq = np.linspace(axis.get_xbound()[0], axis.get_xbound()[1], 100)
    y_seq = a * x_seq + b
    axis.plot(x_seq, y_seq, color="r", linewidth=1.0, linestyle='--')


def main():
    args = parse_args()
    ex_labels = np.load(args.mem_scores_path)['tr_labels']

    dataset = ra_utils.find_dataset_name(args.densities_path)

    acts_arr = np.load(op.join(args.acts_path))
    densities_arr = np.array(fs_utils.read_json(args.densities_path), dtype=np.float32)

    acts_entropies = np.array(fs_utils.read_json(args.class_entropies_path), dtype=np.float32)
    plot_density_dist_relation(densities_arr, acts_arr, acts_entropies, ex_labels, args.out_plot_path, dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('acts_path')
    parser.add_argument('densities_path')
    parser.add_argument('mem_scores_path')
    parser.add_argument('out_plot_path')
    parser.add_argument('class_entropies_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()
