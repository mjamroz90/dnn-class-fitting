"""
CIFAR-10 and CIFAR-100 datasets with support for random labels
"""
from typing import List

import numpy as np
import torchvision
import torchvision.datasets as datasets
import os

from utils.exceptions import RandomizeTestSetException


class CIFARRandomLabels(object):
    """
    Wraps CIFAR datasets in order to randomize train set labels
    """

    def __init__(self, root: str, corrupted_labels_path, corrupt_prob, num_classes, **kwargs):
        """

        Args:
            corrupt_prob: probability of label corruption
            num_classes:
        """
        if not kwargs["train"]:
            raise RandomizeTestSetException()
        super(CIFARRandomLabels, self).__init__(root, **kwargs)
        self.n_classes = num_classes
        if corrupted_labels_path is not None and os.path.exists(corrupted_labels_path):
            if self.__new_torchvision_version():
                self.targets = self.__read_corrupted_labels(corrupted_labels_path)
            else:
                self.train_labels = self.__read_corrupted_labels(corrupted_labels_path)
        elif corrupt_prob > 0 and self.train:
            if self.__new_torchvision_version():
                self.targets = self.__corrupt_labels(corrupt_prob, self.targets)
            else:
                self.train_labels = self.__corrupt_labels(corrupt_prob, self.train_labels)

    def __corrupt_labels(self, corrupt_prob: float, targets: List):
        """

        Args:
            corrupt_prob: probability of label corruption
        """
        labels = np.array(targets)
        np.random.seed(44)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        return labels

    @staticmethod
    def __read_corrupted_labels(path):
        from utils import fs_utils

        corrupted_labels_list = fs_utils.read_json(path)
        return [int(l) for l in corrupted_labels_list]

    @staticmethod
    def __new_torchvision_version():
        return torchvision.__version__ > '0.2.1'


class CIFAR10RandomLabels(CIFARRandomLabels, datasets.CIFAR10):
    """
    Wraps CIFAR-10 dataset in order to randomize train set labels
    """

    def __init__(self, root: str, corrupted_labels_path=None, corrupt_prob=0.0, num_classes=10, **kwargs):
        """

        Args:
            corrupt_prob: probability of label corruption
            num_classes:
        """
        super(CIFAR10RandomLabels, self).__init__(root, corrupted_labels_path, corrupt_prob, num_classes, **kwargs)


class CIFAR100RandomLabels(CIFARRandomLabels, datasets.CIFAR100):
    """
    Wraps CIFAR-100 dataset in order to randomize train set labels
    """

    def __init__(self, root: str, corrupted_labels_path=None, corrupt_prob=0.0, num_classes=100, **kwargs):
        """

        Args:
            corrupt_prob: probability of label corruption
            num_classes:
        """
        super(CIFAR100RandomLabels, self).__init__(root, corrupted_labels_path, corrupt_prob, num_classes, **kwargs)


class CIFAR100ExcludedIndices(datasets.CIFAR100):

    def __init__(self, excluded_indices_path, **kwargs):
        from utils import fs_utils

        super(CIFAR100ExcludedIndices, self).__init__(**kwargs)
        excluded_indices_list = fs_utils.read_json(excluded_indices_path)
        excluded_indices_dedupl = list(set(excluded_indices_list))
        self.excluded_indices = sorted(excluded_indices_dedupl)

        self.data = np.delete(self.data, self.excluded_indices, axis=0)
        self.targets = [t for i, t in enumerate(self.targets) if i not in self.excluded_indices]


class CIFAR100IncludedIndices(datasets.CIFAR100):

    def __init__(self, included_indices, **kwargs):
        from utils import fs_utils

        super(CIFAR100IncludedIndices, self).__init__(**kwargs)
        if isinstance(included_indices, str):
            included_indices_list = fs_utils.read_json(included_indices)
        else:
            included_indices_list = included_indices

        included_indices_dedupl = list(set(included_indices_list))
        self.included_indices = sorted(included_indices_dedupl)
        self.data = self.data[self.included_indices, :, :, :]

        self.targets = [self.targets[index] for index in self.included_indices]


class CIFAR10PklUnsupervised(datasets.CIFAR10):

    def __init__(self, root):
        super().__init__(root, transform=None, train=True, download=True)

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img
