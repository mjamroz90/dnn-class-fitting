import argparse
import os
import os.path as op

import numpy as np
from utils import fs_utils
from utils.logger import log


def list_npy_array_files(in_data_file):
    if op.isdir(in_data_file):
        file_paths = [op.join(in_data_file, p) for p in os.listdir(in_data_file) if p.endswith(('npy',))]
    else:
        file_paths = [in_data_file]

    return file_paths


def cut_sing_vectors_with_sum(s, sum_thresh, dim):
    s2 = np.power(s, 2)
    cum_sums = np.cumsum(s2)
    cum_sums_norm = cum_sums / np.sum(s2)

    if sum_thresh is not None:
        result = np.searchsorted(cum_sums_norm, sum_thresh, side='right')
    else:
        result = cum_sums_norm[dim]
    return result


def transform_input(input_array, vt, trunc_index):
    vt_trunc = vt[:trunc_index, :]

    return np.dot(input_array, vt_trunc.T)


@log
def process_array_files(npy_array_files, axis, num_features, values_thresh, out_file):
    for i, npy_array_file in enumerate(npy_array_files):
        in_array = np.load(npy_array_file)

        if axis == 0:
            in_array = in_array.T

        u, s, vt = np.linalg.svd(in_array)

        if num_features is not None:
            trunc_index = num_features
            projected_array = transform_input(in_array, vt, trunc_index)
            sum_thresh = cut_sing_vectors_with_sum(s, None, trunc_index)
            process_array_files.logger.info("For given num_features: %d, svd thresh is: %.3f" % (num_features,
                                                                                                 sum_thresh))
        else:
            process_array_files.logger.info("Processing array with SVD thresh, SVD values: %s" % str(s))
            trunc_index = cut_sing_vectors_with_sum(s, values_thresh, None)
            projected_array = transform_input(in_array, vt, trunc_index)
            sum_thresh = values_thresh

        process_array_files.logger.info("Processed %d/%d array file, out shape: %s" % (i, len(npy_array_files),
                                                                                       str(projected_array.shape)))

        if len(npy_array_files) > 1:
            i_out_file = op.join(out_file, op.basename(npy_array_file))
            i_out_pkl_file = op.join(out_file, "%s.pkl" % op.splitext(op.basename(npy_array_file))[0])
        else:
            i_out_file = out_file
            i_out_pkl_file = "%s.pkl" % op.splitext(out_file)[0]

        np.save(i_out_file, projected_array)
        fs_utils.write_pickle({'sum_thresh': float(sum_thresh), 'v': vt, 'trunc_index': trunc_index, 's': s},
                              i_out_pkl_file)
        save_sum_thresh_to_metadata_file(float(sum_thresh), i_out_file)


def save_sum_thresh_to_metadata_file(sum_thresh, out_arr_file):
    metadata_file = op.join(op.dirname(out_arr_file), 'metadata.json')
    key_name = op.splitext(op.basename(out_arr_file))[0]
    if op.exists(metadata_file):
        existing_struct = fs_utils.read_json(metadata_file)
        existing_struct[key_name] = sum_thresh
    else:
        existing_struct = {key_name: sum_thresh}

    fs_utils.write_json(existing_struct, metadata_file)


@log
def main():
    args = parse_args()

    if not args.num_features and not args.singular_values_thresh:
        raise ValueError("Specify either --num_features or --singular_values_thresh")

    npy_array_files = list_npy_array_files(args.in_data_file)

    if len(npy_array_files) > 1:
        fs_utils.create_dir_if_not_exists(args.out_file_or_dir)

    process_array_files(npy_array_files, args.axis, args.num_features, args.singular_values_thresh,
                        args.out_file_or_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_data_file')
    parser.add_argument('out_file_or_dir')
    parser.add_argument('--axis', type=int, choices=[0, 1], default=1)
    parser.add_argument('--num_features', type=int)
    parser.add_argument('--singular_values_thresh', type=float)
    return parser.parse_args()


if __name__ == '__main__':
    main()
