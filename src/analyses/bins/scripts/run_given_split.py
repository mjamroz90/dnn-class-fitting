import argparse
import datetime
import logging
import os
import os.path as op
import math

import numpy as np
import mlflow
import torch
from munch import munchify
from torch.utils.data import DataLoader

import base_settings
from datasets.rlcifar import CIFAR100IncludedIndices
from datasets.mini_imagenet import MiniImageNet
from src.transforms import get_train_transform, get_test_transform
from src.experiments import train_cnn
from src.settings import NN_ARCHITECTURES
from src.experiments.settings import SHARED_OPTS
from utils import fs_utils


def train(model, optimizer, scheduler, train_data_loader, val_data_loader, opts, device_obj, init_iter=0,
          val_interval=1):
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(init_iter, opts.epochs, 1):
        if scheduler is not None:
            curr_lr = optimizer.param_groups[0]['lr']
            message = 'Current lr: {lr:.5f}'.format(lr=curr_lr)
            logging.info(message)

        opts.epoch = epoch + 1
        train_metrics = train_cnn.train_epoch(model, train_data_loader, optimizer, scheduler, criterion, opts,
                                              train_cnn.SaveAtEnd(), device_obj)
        if epoch % val_interval == 0:
            test_metrics = train_cnn.validate(model, val_data_loader, criterion, device_obj)
            train_cnn.log_metrics(train_metrics, test_metrics)

    return model


def run_training(train_dataset, val_dataset, opts, device_id, shuffle_train=True):
    logging.info(f'{os.linesep}Running experiments for architecture {opts.model_name}{os.linesep}, '
                 f'dataset: {opts.dataset}, split-index: {opts.split_index}')
    opts.experiment_name = f"{opts.model_name}: running experiment on {opts.dataset} for architecture " \
                           f"{opts.model_name}{os.linesep}, split-index: {opts.split_index}"
    mlflow.set_experiment(opts.experiment_name)

    run_name = f'run-{datetime.datetime.now()}'
    logging.info(f'{os.linesep}Running experiment: {run_name}{os.linesep}')
    device_obj = torch.device("cuda:%d" % device_id if torch.cuda.is_available() else "cpu")

    train_data_loader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers,
                                   shuffle=shuffle_train)
    val_data_loader = DataLoader(val_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers)

    with mlflow.start_run(run_name=run_name):
        train_cnn.log_params(opts)

        model = NN_ARCHITECTURES[opts.model_name](opts.dataset).to(device_obj)
        optimizer = train_cnn.get_optimizer(model.parameters(), opts)
        steps_per_epoch = math.ceil(len(train_dataset) / opts.batch_size)

        logging.info(f"Steps per epoch for dataset, with batch_size={opts.batch_size} is: {steps_per_epoch}")
        scheduler = train_cnn.get_scheduler(train_dataset, optimizer, opts)

        init_iter = 0

        train(model, optimizer, scheduler, train_data_loader, val_data_loader, opts, device_obj,
              init_iter, opts.val_interval)
        mlflow.pytorch.log_model(model, opts.model_name)

    return get_current_experiment_dir(opts)


def fetch_split(splits_arrs, split_index):
    return splits_arrs['train'][split_index], splits_arrs['val'][split_index]


def get_current_experiment_dir(opts):
    from urllib.parse import urlparse

    current_experiment = mlflow.get_experiment_by_name(opts.experiment_name)
    path = urlparse(current_experiment.artifact_location)
    experiment_path = path.path

    return experiment_path


def main():
    args = parse_args()
    splits_arrs = np.load(args.splits_npz_path)

    train_augmentation_transform = get_train_transform(True, args.dataset, True, noise_level=args.noise_level)
    test_transform = get_test_transform(args.dataset, True)

    train_indices, val_indices = fetch_split(splits_arrs, args.split_index)

    if args.dataset == 'cifar100':
        train_dataset = CIFAR100IncludedIndices(included_indices=train_indices, root=base_settings.DATA_ROOT,
                                                download=True, train=True, transform=train_augmentation_transform)
        val_dataset = CIFAR100IncludedIndices(included_indices=val_indices, root=base_settings.DATA_ROOT,
                                              download=True, train=True, transform=test_transform)
    else:
        train_dataset = MiniImageNet(train=True, transform=train_augmentation_transform, indices=train_indices,
                                     reduce_dim=False)
        # Validation dataset here is a train subset chosen validation indices
        val_dataset = MiniImageNet(train=True, transform=test_transform, indices=val_indices, reduce_dim=False)

    SHARED_OPTS['batch_size'] = args.batch_size
    SHARED_OPTS['epochs'] = args.epochs_num
    SHARED_OPTS['dataset'] = args.dataset
    SHARED_OPTS['model_name'] = args.model
    SHARED_OPTS['lr_sched'] = "1e-6:0.4:1e-6:0.15:linear"
    SHARED_OPTS['split_index'] = args.split_index
    SHARED_OPTS['val_interval'] = 10
    SHARED_OPTS['num_workers'] = args.num_workers

    opts = munchify(SHARED_OPTS)
    run_path = run_training(train_dataset, val_dataset, opts, args.device_id)

    if not op.exists(args.out_run_registry_path):
        registry_obj = {}
    else:
        registry_obj = fs_utils.read_json(args.out_run_registry_path)

    registry_obj[args.split_index] = run_path
    fs_utils.write_json(registry_obj, args.out_run_registry_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['cifar100', 'mini_imagenet'])
    parser.add_argument('model', choices=['resnet50', 'ccacnn8x256', 'resnet18'])
    parser.add_argument('splits_npz_path')
    parser.add_argument('split_index', type=int)
    parser.add_argument('out_run_registry_path')
    parser.add_argument('--epochs_num', type=int, default=160)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--noise_level', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=3)
    return parser.parse_args()


if __name__ == '__main__':
    main()
