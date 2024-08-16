import glob
import os
import os.path as osp

from object_centric_video_prediction.data import RobotDataset

PATH = "/home/nfs/inf6/data/datasets/EwertzData/dm_control_4"


class DMControlDataset(RobotDataset):

  def __init__(self, mode, dataset_name, ep_len=500, sample_length=10, random_start=True, img_size=(64, 64)):
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
      paths = list(glob.glob(osp.join(dir_name, '*.png')))
      get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
      paths.sort(key=get_num)
      self.epsisodes.append(paths)
    return
