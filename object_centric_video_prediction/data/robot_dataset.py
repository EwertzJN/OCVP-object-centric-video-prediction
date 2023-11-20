import cv2
import imageio
import numpy as np
from torchvision import transforms
import glob
import os
import os.path as osp
import torch
from PIL import Image

from ..CONFIG import CONFIG

PATH = "/home/nfs/inf6/data/datasets/EwertzData/robot-datasets"

def get_bboxes(seg, max_obj):
    bboxes = []

    for seg_value in np.unique(seg):

        segmentation = np.where(seg == seg_value)

        # Bounding Box
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))

            bboxes.append(torch.tensor([x_min, y_min, x_max, y_max]) / (seg.shape[0]-1))

    while len(bboxes) < max_obj:
        bboxes.append(torch.ones(4) * -1)

    return torch.stack(bboxes)

class RobotDataset:
    """
    DataClass for datasets generated with the visual-block-builder package.

    During training, we sample a random subset of frames in the episode. At inference time,
    we always start from the first frame, e.g., when the ball moves towards the objects, and
    before any collision happens.

    Args:
    -----
    mode: string
        Dataset split to load. Can be one of ['train', 'val', 'test']
    ep_len: int
        Number of frames in an episode. Default is 30
    sample_length: int
        Number of frames in the sequences to load
    random_start: bool
        If True, first frame of the sequence is sampled at random between the possible starting frames.
        Otherwise, starting frame is always the first frame in the sequence.
    """

    def __init__(self, mode, dataset_name, ep_len=30, sample_length=20, random_start=True, img_size=(64, 64), num_slots=None):
        """
        Dataset Initializer
        """
        assert mode in ["train", "val", "valid", "eval", "test"], f"Unknown dataset split {mode}..."
        mode = "val" if mode in ["val", "valid"] else mode
        mode = "test" if mode in ["test", "eval"] else mode
        assert mode in ['train', 'val', 'test'], f"Unknown dataset split {mode}..."

        self.root = os.path.join(PATH, dataset_name, mode)
        self.mode = mode
        self.sample_length = sample_length
        self.random_start = random_start
        self.img_size = img_size
        self.num_slots = num_slots

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()

        # episode-related paramters
        self.epsisodes = []
        self.EP_LEN = ep_len
        if mode == "train" and self.random_start:
            self.seq_per_episode = self.EP_LEN - self.sample_length + 1
        else:
            self.seq_per_episode = 1

        # loading images from data directories and assembling then into episodes
        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '[!{seg}]*.png')))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            self.epsisodes.append(paths)
        return

    def __getitem__(self, index):
        """
        Fetching a sequence from the dataset
        """
        imgs = []
        bboxes = []

        # Implement continuous indexing
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + self.sample_length

        e = self.epsisodes[ep]
        for image_index in range(offset, end):
            img = Image.open(osp.join(e[image_index]))
            img = img.resize(self.img_size)
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)

            seg = imageio.imread("/" + osp.join(*(e[image_index].split("/")[:-1] + [("seg_" + e[image_index].split("/")[-1])])))
            bboxes.append(get_bboxes(seg, self.num_slots))

        img = torch.stack(imgs, dim=0).float()
        bbox = torch.stack(bboxes, dim=0)

        # load actions
        actions = torch.from_numpy(np.load("/" + osp.join(*(self.epsisodes[ep][0].split("/")[:-1] + ["actions.npy"])))[offset:end])

        targets = img
        all_reps = {"videos": img, "bbox_coords": bbox}
        return img, targets, actions, all_reps

    def __len__(self):
        """
        Number of episodes in the dataset
        """
        length = len(self.epsisodes)
        return length


#
