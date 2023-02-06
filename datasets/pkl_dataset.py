from torch.utils import data
from utils import fs_utils

from PIL import Image


class PklDataset(data.Dataset):

    def __init__(self, path, transform):
        self.path = path
        self.ds_pickle = fs_utils.read_pickle(path)

        self.transform = transform

    def __len__(self):
        return len(self.ds_pickle['labels'])

    def __getitem__(self, idx):
        img = self.ds_pickle['data'][idx, :, :, :]
        label = self.ds_pickle['labels'][idx]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = img.transpose((2, 0, 1))

        return img, label
