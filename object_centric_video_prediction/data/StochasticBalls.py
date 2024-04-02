"""
Stochastic bouncing shapes dataset.
Three shapes with random color move on the image grid, bouncing off the image border.
We can add different variants of randomness to the dataset.
"""

import os
import numpy as np 
import cv2
import torch
from torch.utils.data import Dataset
from webcolors import name_to_rgb


class BaseStochasticBalls(Dataset):
    """
    Sotchastic bouncing shapes dataset that generates data on-the-fly.
    This module is used to generate the dataset, which can be then loaded by the module above
    """

    MOVING_SPECS = {
        "speed_min": 2,
        "speed_max": 5,
        "acc_min": 0,
        "acc_max": 0
    }
    COLORS = {
        0: "red",
        1: "cyan",
        2: "green",
        3: "blue",
        4: "magenta",
        5: "yellow",
        6: "orange",
        7: "purple",
        8: "white",
        9: "brown"
    }
    SHAPE = {
        0: "ball",
        1: "triangle",
        2: "square"
    }
    PROPERTIES = {
        "shape": {
            "type": "categorical",
            "num_values": len(SHAPE.keys())
        },
        "color": {
            "type": "categorical",
            "num_values": len(COLORS.keys())
        },
        "positions": {
            "type": "continuous",
            "num_values": 2
        },
        "velocities": {
            "type": "temporal",
            "num_values": 2
        }
    }

    def __init__(self, num_frames=30, num_balls=3, img_size=64, random_bounce=False, 
                 random_reroll=False, random_reroll_frames=5, reroll_velocity=True,
                 reroll_direction=False):
        """
        Initializer of the Stochastic Balls dataset
        """
        self.num_frames = num_frames
        self.num_balls = num_balls
        self.img_size = img_size
        self.random_reroll = random_reroll
        self.random_reroll_frames = random_reroll_frames
        self.random_bounce = random_bounce

        self.reroll_velocity = reroll_velocity
        self.reroll_direction = reroll_direction        
        self.ball_size = 20

        # loading moving parameters
        speed_max, acc_max = self.MOVING_SPECS["speed_max"], self.MOVING_SPECS["acc_max"]
        speed_min, acc_min = self.MOVING_SPECS["speed_min"], self.MOVING_SPECS["acc_min"]

        self.get_speed = lambda: np.random.randint(speed_min, speed_max+1)
        self.get_acc = lambda: np.random.randint(acc_min, acc_max+1)
        self.get_init_pos = lambda img_size, digit_size: np.random.randint(0, img_size-digit_size)
        self.get_direction = lambda: np.random.choice([-1, 1])
        self.get_color = lambda: np.random.choice(list(self.COLORS.values()))
        self.get_shape = lambda: np.random.choice(list(self.SHAPE.values()))
        return

    def __len__(self):
        """ """
        raise NotImplementedError("Base class does not implement __len__ functionality")

    def __getitem__(self, idx):
        """
        Sampling sequence
        """
        raise NotImplementedError("Base class does not implement __getitem__ functionality")
    
    def generate_new_seq(self):
        """
        Generating a new sequence of the datasets
        """
        frames = np.zeros((self.num_frames, 3, self.img_size, self.img_size))
        speeds_per_frame = torch.empty(self.num_frames, self.num_balls, 2)
        pos_per_frame = torch.empty(self.num_frames, self.num_balls, 2)

        # initial conditions
        shapes, next_poses, speeds, obj_metas = [], [], [], []
        for i in range(self.num_balls):
            shape, pos, speed, meta = self._sample_shape()
            meta["depth"] = self.num_balls - i - 1
            shapes.append(shape)
            next_poses.append(pos)
            speeds.append(speed)
            obj_metas.append(meta)

        # generating sequence by moving the shapes given velocity
        for i, frame in enumerate(frames):
            for j, (digit, cur_pos, speed) in enumerate(zip(shapes, next_poses, speeds)):
                if not self.random_bounce and self.random_reroll:
                    speed = self._check_speed_reroll(i, speed)
                digit_size = digit.shape[-2]
                speed, cur_pos = self._move_shape(
                        speed=speed,
                        cur_pos=cur_pos,
                        img_size=self.img_size,
                        digit_size=digit_size
                    )
                speeds[j] = speed
                next_poses[j] = cur_pos

                idx = digit.sum(dim=0) > 0
                frame[:, cur_pos[0]:cur_pos[0]+digit_size, cur_pos[1]:cur_pos[1]+digit_size][:, idx] = digit[:, idx]

                speeds_per_frame[i, j, 0], speeds_per_frame[i, j, 1] = speed[0], speed[1]
                pos_per_frame[i, j, 0], pos_per_frame[i, j, 1] = cur_pos[0], cur_pos[1]
            frames[i] = np.clip(frame, 0, 1)
        frames = torch.Tensor(frames)
        
        motion_metas = {
            "positions": pos_per_frame,
            "speeds": speeds_per_frame,
        }
        return frames, obj_metas, motion_metas

    def _check_speed_reroll(self, frame_idx, speed):
        """
        Computing a new random velocity, and potentially direction, every few
        time steps
        """
        # no reroll
        if not (self.reroll_velocity or self.reroll_direction):
            return speed
        if not (frame_idx % self.random_reroll_frames == self.random_reroll_frames-1):
            return speed 

        new_speed = self._sample_speed()
        if self.reroll_velocity and self.reroll_direction:
            speed = new_speed
        elif self.reroll_velocity:
            # Set the new speed, keeping the sign the same.
            speed[0] = np.sign(speed[0])*abs(new_speed[0])
            speed[1] = np.sign(speed[1])*abs(new_speed[1])
        elif self.reroll_direction:
            # Maintain the speed, set new direction
            # If the signs are the same, this will cancel and do nothing
            speed[0] *= np.sign(speed[0])*np.sign(new_speed[0])
            speed[1] *= np.sign(speed[1])*np.sign(new_speed[1])

        return speed

    def _sample_shape(self):
        """
        Sampling shape, original position and speed
        """
        shape_name = self.get_shape()
        color_name = self.get_color()
        shape = self._make_shape(shape_name=shape_name, color_name=color_name)

        # obtaining position in original frame
        x_coord = self.get_init_pos(self.img_size, self.ball_size)
        y_coord = self.get_init_pos(self.img_size, self.ball_size)
        cur_pos = np.array([y_coord, x_coord])

        speed = self._sample_speed()

        meta = {
            "color": color_name,
            "shape": shape_name,
            "init_speed": speed,
            "init_pos": torch.from_numpy(cur_pos)
        }
        return shape, cur_pos, speed, meta

    def _sample_speed(self):
        """
        Sample a new speed
        """
        speed_x = self.get_speed() * self.get_direction()
        speed_y = self.get_speed() * self.get_direction()
        speed = np.array([speed_y, speed_x])
        return speed

    def _move_shape(self, speed, cur_pos, img_size, digit_size):
        """
        Performing a shape movement. Also producing bounce and making appropriate changes
        """
        next_pos = cur_pos + speed        
        if next_pos[0] < 0:
            next_pos[0] = 0
            if not self.random_bounce:
                speed[0] = -1 * speed[0]
            else:
                speed[0] = self.get_speed() * -1 * np.sign(speed[0])
                speed[1] = self.get_speed() * self.get_direction()
        elif next_pos[0] > img_size - digit_size:
            next_pos[0] = img_size - digit_size - 1
            if not self.random_bounce:
                speed[0] = -1 * speed[0]
            else:
                speed[0] = self.get_speed() * -1 * np.sign(speed[0])
                speed[1] = self.get_speed() * self.get_direction()
            
        if next_pos[1] < 0:
            next_pos[1] = 0
            if not self.random_bounce:
                speed[1] = -1 * speed[1]
            else:
                speed[1] = self.get_speed() * -1 * np.sign(speed[1])
                speed[0] = self.get_speed() * self.get_direction()
        elif next_pos[1] > img_size - digit_size:
            next_pos[1] = img_size - digit_size - 1
            if not self.random_bounce:
                speed[1] = -1 * speed[1]
            else:
                speed[1] = self.get_speed() * -1 * np.sign(speed[1])
                speed[0] = self.get_speed() * self.get_direction()
        return speed, next_pos

    def _make_shape(self, shape_name, color_name):
        """ """
        aux = torch
        aux = np.zeros((21, 21))
        if shape_name == "ball":
            shape = cv2.circle(aux, (10, 10), int(10), 1, -1)
        elif shape_name == "square":
            shape = cv2.rectangle(aux, (0,0), (21, 21), 1, -1)
        elif shape_name == "triangle":
            coords = np.array([[10, 0], [0,21], [21,21]])
            coords = coords.reshape((-1, 1, 2))
            shape = cv2.fillPoly(aux, [coords], 255, 1)
        else:
            raise ValueError(f"Unkwnown {shape_name = }...")
        shape = torch.Tensor(shape).unsqueeze(0).repeat(3, 1, 1)
        color = torch.tensor(name_to_rgb(color_name)).float() / 255
        shape = (shape * color.view(-1, 1, 1))
        return shape


class GenerateStochasticBalls(BaseStochasticBalls):
    """
    Sotchastic bouncing shapes dataset that generates data on-the-fly for the training set
    and loads the precomputed data from the test/validation sets.
    """

    def __len__(self):
        """ """
        return 10000

    def __getitem__(self, idx):
        """
        Sampling sequence
        """
        frames, obj_metas, motion_metas = self.generate_new_seq()
        meta = {
            "object_anns": obj_metas,
            "motion_anns": motion_metas
        }
        return frames, frames, meta


class StochasticBalls(BaseStochasticBalls):
    """
    Sotchastic bouncing shapes dataset that generates data on-the-fly.
    This module is used to generate the dataset, which can be then loaded by the module above
    """
    
    VARIANTS = {
        "deterministic": "BouncingShapes",
        "random_bounce": "StochasticShapes_bounce",
        "random_reroll": "StochasticShapes_step",
    }
    
    def __init__(self,  datapath, split, variant, num_frames=30, num_balls=3, img_size=64,
                 random_bounce=True, random_reroll=False, random_reroll_frames=5,
                 reroll_velocity=True, reroll_direction=False, **kwargs):
        """
        Dataset initializer
        """
        assert variant in self.VARIANTS, f"Unknown {variant = }. Use one of {self.VARIANTS.keys() = }..."
        assert split in ["train", "val", "valid", "eval", "test"], f"Unknown dataset split {split}..."
        split = "val" if split in ["val", "valid"] else split
        split = "test" if split in ["test", "eval"] else split
        assert split in ['train', 'val', 'test'], f"Unknown dataset split {split}..."
        self.split = split
        self.variant = variant
        self.datapath = os.path.join(datapath, self.VARIANTS[variant])
        assert os.path.exists(self.datapath)
        
        super().__init__(
            num_frames=num_frames,
            num_balls=num_balls,
            img_size=img_size,
            random_bounce=random_bounce,
            random_reroll=random_reroll,
            random_reroll_frames=random_reroll_frames,
            reroll_velocity=reroll_velocity,
            reroll_direction=reroll_direction,
        )
        self._load_precomputed_data()
        return
    
    def _load_precomputed_data(self):
        """ Loading precomputed validation set """
        if self.split == "train":
            return
        data_file = "train_data.pt"  if self.split == "train" else "valid_data.pt"
        anns_file = "train_annotations.pt"  if self.split == "train" else "valid_annotations.pt"
        print(f"    Loading validation data")
        self.data = torch.load(os.path.join(self.datapath, data_file))
        print(f"    Loading validation annotations")
        anns = torch.load(os.path.join(self.datapath, anns_file))
        self.object_anns = anns["objects"]
        self.motion_anns = anns["motion"]
        return

    def __len__(self):
        """
        Number of sequences. During training we set 10000 by hand
        """
        num_seqs = 10000 if self.split == "train" else len(self.data)
        return num_seqs

    def __getitem__(self, i):
        """
        Sampling sequence:
          - Training: Generate a new random sequence
          - Testing:  Use the precomputed sequences
        """
        if self.split == "train":
            frames, obj_metas, motion_metas = self.generate_new_seq()
        else:
            frames = self.data[i][:self.num_frames]
            obj_metas = self.object_anns[i]
            motion_metas = self.motion_anns[i]
            motion_metas["positions"] = motion_metas["positions"][:self.num_frames]
            motion_metas["speeds"] = motion_metas["speeds"][:self.num_frames]
            
        # converting object properties labels to IDs
        colors, shapes = [], []
        for i, cur_obj_meta in enumerate(obj_metas):
            cur_colors = list(self.COLORS.values()).index(cur_obj_meta["color"]) 
            cur_shapes = list(self.SHAPE.values()).index(cur_obj_meta["shape"])
            colors.append(torch.tensor(cur_colors))
            shapes.append(torch.tensor(cur_shapes))
           
        meta = {
            "color": torch.stack(colors),
            "shape": torch.stack(shapes),
            "positions": motion_metas["positions"] / 43,  # normalized to [0, 1]
            "velocities": motion_metas["speeds"] / self.MOVING_SPECS["speed_max"] # normalized to [-1, 1]
        }
        return frames, frames, meta
        
