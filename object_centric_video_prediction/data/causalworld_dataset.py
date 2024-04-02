import math

import numpy as np
from torch.utils.data import Sampler
from torchvision import transforms
import glob
import os
import os.path as osp
import torch
from PIL import Image


PATH = "/home/nfs/inf6/data/datasets/EwertzData/causalworld/CwTargetEnv-N4C11S1S1-Tr1000000-Val10000"


class CausalworldDataset:
    """
    DataClass for a dataset generated with the causalworld environment as used in https://github.com/jsikyoon/OCRL/tree/main/envs.

    During training, we sample a random subset of frames in the episode. At inference time,
    we always start from the first frame.

    Args:
    -----
    mode: string
        Dataset split to load. Can be one of ['train', 'val']
    sample_length: int
        Number of frames in the sequences to load
    random_start: bool
        If True, first frame of the sequence is sampled at random between the possible starting frames.
        Otherwise, starting frame is always the first frame in the sequence.
    """

    def __init__(self, mode, img_size=(64, 64)):
        """
        Dataset Initializer
        """
        assert mode in ["train", "val", "valid"], f"Unknown dataset split {mode}..."
        mode = "val" if mode in ["val", "valid"] else mode
        assert mode in ['train', 'val'], f"Unknown dataset split {mode}..."

        self.root = os.path.join(PATH, 'TrainingSet' if mode == 'train' else 'ValidationSet')
        self.mode = mode
        self.img_size = img_size

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()

        # loading images from data directories and assembling then into episodes
        self.epsisodes = []
        self.seq_length = []
        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            self.epsisodes.append(paths)
            self.seq_length.append(len(paths))
        self.seq_length = np.array(self.seq_length)

        return

    def __getitem__(self, index):
        """
        Fetching a sequence from the dataset
        """
        imgs = []

        ep, sample_length = index
        e = self.epsisodes[ep]
        offset = np.random.randint(0, self.seq_length[ep] - sample_length) if self.seq_length[ep] - sample_length > 0 else 0
        end = offset + sample_length

        for image_index in range(offset, end):
            img = Image.open(osp.join(e[image_index]))
            img = img.resize(self.img_size)
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)
        img = torch.stack(imgs, dim=0).float()

        # load actions
        actions = torch.from_numpy(np.load("/" + osp.join(*(self.epsisodes[ep][0].split("/")[:-1] + ["actions.npy"])))[offset:end-1])

        targets = img
        all_reps = {"videos": img}
        return img, targets, actions, all_reps

    def __len__(self):
        """
        Number of episodes in the dataset
        """
        length = len(self.epsisodes)
        return length


class VariableSeqLengthBatchSampler(Sampler):

    def __init__(self, dataset, batch_size, max_sample_length, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_sample_length = max_sample_length
        self.shuffle = shuffle

    def __iter__(self):
        index_sequence = np.arange(len(self.dataset)) if not self.shuffle else np.random.permutation(np.arange(len(self.dataset)))
        batch_sequence = np.split(index_sequence, np.arange(self.batch_size, len(index_sequence), self.batch_size))

        for batch_indices in batch_sequence:
            seq_length = np.min(np.append(self.dataset.seq_length[batch_indices], self.max_sample_length))

            batch = [[idx, seq_length] for idx in batch_indices]

            yield batch

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

#
