import argparse
import os
import os.path as op

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import cifar
from datasets import rlcifar
import numpy as np

import base_settings
from src.transforms import get_test_transform
from datasets.mini_imagenet import MiniImageNet
from datasets.pkl_dataset import PklDataset
from utils import fs_utils
from utils.logger import log

BATCH_SIZE = 128
NUM_WORKERS = 8

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@log
def get_test_dataset_loader(dataset, normalize_data=True, use_trainset=False, noise_level=0.0,
                            excluded_indices_path=None, included_indices_path=None, batch_size=BATCH_SIZE):
    test_transform = get_test_transform(dataset, normalize_data, noise_level)
    get_test_dataset_loader.logger.info("Noise level: %f" % noise_level)

    if dataset == 'cifar10':
        test_dataset = cifar.CIFAR10(base_settings.DATA_ROOT, download=True, train=use_trainset,
                                     transform=test_transform)
    elif dataset == 'cifar100':
        if excluded_indices_path is not None:
            test_dataset = rlcifar.CIFAR100ExcludedIndices(excluded_indices_path, root=base_settings.DATA_ROOT,
                                                           download=True, train=use_trainset, transform=test_transform)
        elif included_indices_path is not None:
            test_dataset = rlcifar.CIFAR100IncludedIndices(included_indices_path, root=base_settings.DATA_ROOT,
                                                           download=True, train=use_trainset, transform=test_transform)
        else:
            test_dataset = cifar.CIFAR100(base_settings.DATA_ROOT, download=True, train=use_trainset,
                                          transform=test_transform)

    elif dataset == 'mini_imagenet':
        test_dataset = MiniImageNet(train=use_trainset, transform=test_transform, reduce_dim=False)
    else:
        assert op.exists(dataset)
        test_dataset = PklDataset(dataset, test_transform)

    test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)
    return test_dataset_loader


def get_models_paths(model_file_path):
    if op.isdir(model_file_path):
        model_file_names = [p for p in os.listdir(model_file_path) if p.endswith(('pth',))]
        model_file_names = sorted(model_file_names, key=lambda x: int(x.split('.')[0].split('_')[1]))
        model_paths = [op.join(model_file_path, p) for p in model_file_names]
    else:
        model_paths = [model_file_path]

    return model_paths


def process_feature_map(processing_type, output):
    from torch import nn

    if processing_type == 'avg_pooling':
        feature_map_h = output.shape[2]
        process_op = nn.AvgPool2d(kernel_size=feature_map_h)
    elif processing_type == 'max_pooling':
        feature_map_h = output.shape[2]
        process_op = nn.MaxPool2d(kernel_size=feature_map_h)
    else:
        raise ValueError("Only available processing_type is ['avg_pooling', 'max_pooling']")
    processing_out = torch.squeeze(process_op(output))

    return processing_out


@log
def do_predictions_with_model(dataset_loader, model, model_type, layers_indices,
                              processing_type, noise_samples=1, vt=None):
    activations = {i: [] for i in layers_indices}

    def get_activation(li):
        def hook(model, input, output):
            processed_out = process_feature_map(processing_type, output.data)
            processed_out = processed_out.cpu().numpy()
            if vt is not None:
                processed_out = np.dot(processed_out, vt[li].T)
            activations[li].append(processed_out)

        return hook

    assert(model_type in ['resnet', 'ccacnn', 'ccacnn_dropout'])

    if model_type == 'resnet':
        if 0 in layers_indices:
            model.layer1.register_forward_hook(get_activation(0))
        if 1 in layers_indices:
            model.layer2.register_forward_hook(get_activation(1))
        if 2 in layers_indices:
            model.layer3.register_forward_hook(get_activation(2))
        if 3 in layers_indices:
            model.layer4.register_forward_hook(get_activation(3))
    else:
        for index in layers_indices:
            # After conv
            abs_layer_index = (index * 4) if model_type == 'ccacnn_dropout' else (index * 3)
            # abs_layer_index = (index * 4 + 2) if model_type == 'ccacnn_dropout' else (index * 3 + 2) - after ReLU
            model.layers[abs_layer_index].register_forward_hook(get_activation(index))

    with torch.no_grad():
        for sample in range(noise_samples):
            if noise_samples > 1:
                do_predictions_with_model.logger.info("Sample: %d" % sample)
            for i, (x_train, _) in enumerate(dataset_loader):
                x_train = x_train.to(DEVICE)

                model(x_train)
                do_predictions_with_model.logger.info("Predicted %d batch" % i)

    activations = {i: np.vstack(i_acts) for i, i_acts in activations.items()}
    if noise_samples > 1:
        activations = {i: i_acts.reshape((noise_samples, i_acts.shape[0] // noise_samples, -1)).transpose(1, 0, 2)
                       for i, i_acts in activations.items()}

    do_predictions_with_model.logger.info("Finished predicting, activations shape: %s" %
                                          str({i: v.shape for i, v in activations.items()}))

    return activations


@log
def do_all_predictions_and_aggregate(models_paths, predict_func):
    activations = {}
    for i, mp in enumerate(models_paths):
        model = torch.load(mp, map_location=DEVICE)

        # model_activations: dict - {layer_index -> [10k, out_filters]}
        model.eval()
        model_activations = predict_func(model)
        if len(activations) == 0:
            activations = {li: [li_acts] for li, li_acts in model_activations.items()}
        else:
            activations = {li: activations[li] + [model_activations[li]] for li in model_activations}

        do_all_predictions_and_aggregate.logger.info("Made predictions with %d/%d model" % (i, len(models_paths)))

    return activations


def choose_layers_indices(model_name):
    if model_name.startswith(('resnet',)):
        layers_num = 4
    else:
        if 'dropout' in model_name:
            model_name = model_name.split('_')[0]

        model_name_no_prefix = model_name[6:]
        if len(model_name) == 6 or model_name_no_prefix[0] == '1':
            layers_num = 11
        elif model_name_no_prefix[0] == '8':
            layers_num = 8
        else:
            raise ValueError("Cannot parse model name: %s" % model_name)

    return [int(i) for i in range(layers_num)]


@log
def load_svd(svd_root, feature_map_processing, layer_indices, target_dims):
    load_svd.logger.info("Target dimmensionality: %s" % target_dims)

    vt = {}
    for l in layer_indices:
        svd_file = op.join(svd_root, '%s_%d_acts_ld.pkl' % (feature_map_processing, l))
        svd_base = fs_utils.read_pickle(svd_file)['v']
        vt[l] = svd_base[:target_dims[l], :]
        load_svd.logger.info("Loaded SVD base from: %s" % svd_file)

    return vt


@log
def main():
    args = parse_args()

    if (args.reduce_dim is not None) and (args.svd_root is None):
        raise ValueError("Missing svd_root argument")

    if args.excluded_indices_path is not None and args.included_indices_path is not None:
        raise ValueError("excluded_indices_path and included_indices_path cannot be simultaneously non-empty")

    fs_utils.create_dir_if_not_exists(args.out_dir)
    normalize_data = False if args.no_normalize else True

    if args.noise_level > 0.0:
        dataset_loader = get_test_dataset_loader(args.dataset, normalize_data,
                                                 args.use_trainset, args.noise_level,
                                                 excluded_indices_path=args.excluded_indices_path,
                                                 included_indices_path=args.included_indices_path)
    else:
        dataset_loader = get_test_dataset_loader(args.dataset, normalize_data,
                                                 args.use_trainset, excluded_indices_path=args.excluded_indices_path,
                                                 included_indices_path=args.included_indices_path)
    choose_subset_func = lambda arr: arr

    if args.model_name.startswith(('resnet',)):
        model_type = 'resnet'
    elif args.model_name.endswith('_dropout'):
        model_type = 'ccacnn_dropout'
    else:
        model_type = 'ccacnn'

    layers_indices = choose_layers_indices(args.model_name)
    models_paths = get_models_paths(args.model_file_path)

    if args.reduce_dim is not None:
        target_dims = [int(d) for d in args.reduce_dim.split(',')]
        for d in target_dims:
            if d <= 0: raise ValueError("reduce_dim must be all > 0")

        vt = load_svd(args.svd_root, args.feature_map_processing,
                      layers_indices, target_dims)
    else:
        vt = None

    main.logger.info("PyTorch device detected: %s" % DEVICE)

    layers_out_activations = do_all_predictions_and_aggregate(models_paths,
                                                              lambda m: do_predictions_with_model(
                                                                  dataset_loader, m,
                                                                  model_type, layers_indices,
                                                                  args.feature_map_processing,
                                                                  args.noise_samples,
                                                                  vt))

    for li, li_out_activations in layers_out_activations.items():
        if vt is None:
            out_file = "%s_%d_acts" % (args.feature_map_processing, li)
        else:
            out_file = "%s_%d_acts_ld" % (args.feature_map_processing, li)
        if args.agg_mode == 'aggregate' or args.agg_mode == 'both':
            out_activations = np.hstack(tuple(li_out_activations))
            acts_to_save = choose_subset_func(out_activations)
            np.save(op.join(args.out_dir, "%s.npy" % out_file), acts_to_save)
        if args.agg_mode == 'dump_all' or args.agg_mode == 'both':
            out_dir = op.join(args.out_dir, out_file)
            fs_utils.create_dir_if_not_exists(out_dir)
            for i, i_model_acts in enumerate(li_out_activations):
                acts_to_save = choose_subset_func(i_model_acts)
                model_index = int(op.splitext(op.basename(models_paths[i]))[0].split('_')[-1])
                np.save(op.join(out_dir, "model_%d.npy" % model_index), acts_to_save)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', choices=['ccacnn', 'ccacnn_dropout',
                                               'ccacnn11x128', 'ccacnn11x128_dropout',
                                               'ccacnn11x192', 'ccacnn11x192_dropout',
                                               'ccacnn11x256', 'ccacnn11x256_dropout',
                                               'ccacnn11x384', 'ccacnn11x384_dropout',
                                               'ccacnn8x128', 'ccacnn8x128_dropout',
                                               'ccacnn8x192', 'ccacnn8x192_dropout',
                                               'ccacnn8x256', 'ccacnn8x256_dropout',
                                               'resnet50', 'resnet18'])
    parser.add_argument('model_file_path')
    parser.add_argument('dataset')
    parser.add_argument('out_dir')
    parser.add_argument('--feature_map_processing', choices=['avg_pooling', 'max_pooling'], required=True)
    parser.add_argument('--noise_level', type=float, default=0.0)
    parser.add_argument('--noise_samples', type=int, default=1)
    parser.add_argument('--reduce_dim')
    parser.add_argument('--svd_root')
    parser.add_argument('--agg_mode', choices=['aggregate', 'dump_all', 'both'], default='dump_all')
    parser.add_argument('--no_normalize', action='store_true')
    parser.add_argument('--use_trainset', action='store_true')
    parser.add_argument('--excluded_indices_path')
    parser.add_argument('--included_indices_path')

    return parser.parse_args()


if __name__ == '__main__':
    main()
