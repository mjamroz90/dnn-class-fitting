import argparse

import pandas as pd

from utils import fs_utils
from utils import logger
from src.analyses.scripts.results_analysis import res_analysis_utils as ra_utils

Y_LABEL = 'F1 Score'
X_LABEL = 'Stage number'


def generate_pd_data_frame(class_results_dict, keys):
    layer_index_title, layer_f1_title, examples_title, traces_title = X_LABEL, Y_LABEL, 'Examples', 'Trace'
    data = {layer_index_title: [], layer_f1_title: [], examples_title: [], traces_title: []}
    examples_type_to_series = {'most_memorized': 'most memorized', 'least_memorized': 'least memorized',
                               'random': 'random'}
    for layer_index, key in enumerate(keys):
        for examples_type, examples_stats in class_results_dict[key].items():
            for trace_index, trace_stats in enumerate(examples_stats):
                data[layer_index_title].append(layer_index + 1)
                data[examples_title].append(examples_type_to_series[examples_type])
                data[traces_title].append(trace_index)
                data[layer_f1_title].append(trace_stats['f1'])

    return pd.DataFrame(data=data), {'x': layer_index_title, 'y': layer_f1_title,
                                     'hue': examples_title, 'palette': ra_utils.EXAMPLES_PALETTE}

def plot_with_seaborn(data_frame, plot_args, out_plot_file, title):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style='darkgrid')
    sns.set_context('paper')
    plt.tight_layout()

    line_plot = sns.barplot(data=data_frame, **plot_args, ci='sd')
    line_plot.set_xticks([1, 2, 3, 4])
    line_plot.get_legend().set_title(None)
    line_plot.set_title(title, fontsize=ra_utils.TITLE_FONT_SIZE)
    line_plot.set_xlabel(X_LABEL, fontsize=ra_utils.AXES_FONT_SIZE)
    line_plot.set_ylabel(Y_LABEL, fontsize=ra_utils.AXES_FONT_SIZE)
    line_plot.figure.savefig(out_plot_file)


@logger.log
def main():
    args = parse_args()
    class_results_dict = fs_utils.read_json(args.class_results_json_path)
    keys = sorted([k for k in class_results_dict if k.split('_')[0] == args.acts_type],
                  key=lambda k: int(k.split('_')[2]))
    main.logger.info("Will process following keys: %s" % str(keys))
    plot_df, plot_args = generate_pd_data_frame(class_results_dict, keys)

    plot_with_seaborn(plot_df, plot_args, args.out_png_plot_path, ra_utils.find_dataset_name(
        args.class_results_json_path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('class_results_json_path')
    parser.add_argument('out_png_plot_path')
    parser.add_argument('--acts_type', choices=['avg', 'max'], required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
