import logging
import argparse

import mlflow
from munch import munchify
from torchvision.datasets import CIFAR10, CIFAR100
import os
import os.path as op
import numpy as np

import base_settings
from datasets.rlcifar import CIFAR10RandomLabels, CIFAR100RandomLabels, CIFAR100ExcludedIndices
from datasets.mini_imagenet import MiniImageNet
from src.experiments.settings import SHARED_OPTS
from src.experiments.train_cnn import run_experiment
from src.transforms import get_train_transform, get_test_transform


def run_experiments(datasets, opts, architecture_name, transforms):
    train_true_dataset, train_random_dataset, test_dataset = datasets
    train_transform, train_augmentation_transform = transforms

    logging.info(f'{os.linesep}Running experiments for architecture {architecture_name}{os.linesep}')

    opts.model_name = architecture_name

    if opts.types is None or 'true_aug' in opts.types:
        opts.experiment_name = f'{architecture_name} True Labels with Augmentation'
        mlflow.set_experiment(opts.experiment_name)

        exp_existing_count = get_already_existing_runs_and_clean(opts.experiment_name)
        logging.info(f'Number of runs existing for experiment {opts.experiment_name}: {exp_existing_count}')

        for i in range(opts.runs_count - exp_existing_count):
            run_experiment(train_true_dataset, test_dataset, opts)

    train_true_dataset.transform = train_transform

    if opts.types is None or 'true' in opts.types:
        opts.experiment_name = f'{architecture_name} True Labels w/o Augmentation'
        mlflow.set_experiment(opts.experiment_name)

        exp_existing_count = get_already_existing_runs_and_clean(opts.experiment_name)
        logging.info(f'Number of runs existing for experiment {opts.experiment_name}: {exp_existing_count}')

        for i in range(opts.runs_count - exp_existing_count):
            run_experiment(train_true_dataset, test_dataset, opts)

    if not architecture_name.endswith('_dropout'):
        if opts.types is None or 'random' in opts.types:
            opts.experiment_name = f'{architecture_name} Random Labels'

            mlflow.set_experiment(opts.experiment_name)

            exp_existing_count = get_already_existing_runs_and_clean(opts.experiment_name)
            logging.info(f'Number of runs existing for experiment {opts.experiment_name}: {exp_existing_count}')

            for i in range(opts.runs_count - exp_existing_count):
                run_experiment(train_random_dataset, test_dataset, opts)

    train_true_dataset.transform = train_augmentation_transform


def get_already_existing_runs_and_clean(experiment_name):
    from urllib.parse import urlparse
    import shutil

    current_experiment = mlflow.get_experiment_by_name(experiment_name)
    path = urlparse(current_experiment.artifact_location)
    experiment_path = path.path

    runs_dirs = [op.join(experiment_path, p) for p in os.listdir(experiment_path)
                 if op.isdir(op.join(experiment_path, p))]
    non_empty_count = 0

    for run_dir in runs_dirs:
        if not os.listdir(op.join(run_dir, 'artifacts')):
            shutil.rmtree(run_dir)
        else:
            non_empty_count += 1

    return non_empty_count


def main():
    args = parse_args()

    normalize_data = False if args.no_normalize else True

    train_transform = get_train_transform(False, args.dataset, normalize_data)
    train_augmentation_transform = get_train_transform(True, args.dataset, normalize_data, noise_level=args.noise_level)
    test_transform = get_test_transform(args.dataset, normalize_data)

    if args.dataset == 'cifar10':
        train_random_dataset = CIFAR10RandomLabels(root=base_settings.DATA_ROOT,
                                                   corrupted_labels_path=base_settings.CIFAR10_CORRUPTED_LABELS_PATH,
                                                   transform=train_transform, train=True, download=True)
        train_true_dataset = CIFAR10RandomLabels(root=base_settings.DATA_ROOT, corrupt_prob=0.,
                                                 transform=train_augmentation_transform, train=True, download=True)
        test_dataset = CIFAR10(base_settings.DATA_ROOT, download=True, train=False, transform=test_transform)
    elif args.dataset == 'cifar100':
        train_random_dataset = CIFAR100RandomLabels(root=base_settings.DATA_ROOT,
                                                    corrupted_labels_path=base_settings.CIFAR100_CORRUPTED_LABELS_PATH,
                                                    transform=train_transform, train=True, download=True)
        if not args.excluded_indices_path:
            train_true_dataset = CIFAR100RandomLabels(root=base_settings.DATA_ROOT, corrupt_prob=0.,
                                               transform=train_augmentation_transform, train=True, download=True)
        else:
            train_true_dataset = CIFAR100ExcludedIndices(excluded_indices_path=args.excluded_indices_path,
                                                         root=base_settings.DATA_ROOT,
                                                         transform=train_augmentation_transform,
                                                         train=True, download=True)

        test_dataset = CIFAR100(base_settings.DATA_ROOT, download=True, train=False, transform=test_transform)
    else:
        train_random_dataset = MiniImageNet(train=True, transform=train_transform,
                                            corrupted_labels_path=base_settings.MINI_IMAGENET_CORRUPTED_LABELS_PATH)
        train_true_dataset = MiniImageNet(train=True, transform=train_augmentation_transform, reduce_dim=False)
        test_dataset = MiniImageNet(train=False, transform=test_transform, reduce_dim=False)

    SHARED_OPTS['batch_size'] = args.batch_size
    SHARED_OPTS['lr_sched'] = args.lr_sched
    SHARED_OPTS['epochs'] = args.epochs_num
    SHARED_OPTS['runs_count'] = args.runs_count
    SHARED_OPTS['dataset'] = args.dataset
    SHARED_OPTS['types'] = args.types
    opts = munchify(SHARED_OPTS)

    datasets = train_true_dataset, train_random_dataset, test_dataset
    transforms = train_transform, train_augmentation_transform

    np.random.seed(args.seed)

    for net_name in args.network_names:
        run_experiments(datasets, opts, net_name, transforms)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('network_names', nargs='+')
    parser.add_argument('seed', type=int)
    parser.add_argument('dataset', choices=['cifar10', 'cifar100', 'imagenet', 'mini_imagenet'])
    parser.add_argument('--batch_size', type=int, default=SHARED_OPTS['batch_size'])
    parser.add_argument('--lr_sched', default="1e-6:0.4:1e-6:0.15:linear")
    parser.add_argument('--epochs_num', type=int, default=SHARED_OPTS['epochs'])
    parser.add_argument('--runs_count', type=int, default=SHARED_OPTS['runs_count'])
    parser.add_argument('--types', nargs='+', choices=['true', 'true_aug', 'random'])
    parser.add_argument('--no_normalize', action='store_true')
    parser.add_argument('--excluded_indices_path')
    parser.add_argument('--noise_level', type=float, default=0.0)
    return parser.parse_args()


if __name__ == '__main__':
    main()
