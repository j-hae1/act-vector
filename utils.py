import numpy as np
import torch
import os
import h5py
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from glob import glob
from copy import deepcopy

import IPython
e = IPython.embed

from utils_robot import PoseToKeyPoint, get_relative_robot_pose, get_relative_robot_tip_position
from torch_jit_utils import  unscale_transform, scale_transform

def pad_and_stack_with_mask(tensor_list):
    # Find the maximum length among the tensors
    max_length = max(tensor.size(0) for tensor in tensor_list)
    
    # Determine the shape for the padded tensors
    batch_size = len(tensor_list)
    feature_size = tensor_list[0].size(1)
    
    # Initialize an empty tensor to hold the padded tensors
    padded_tensors = torch.zeros(batch_size, max_length, feature_size, device=tensor_list[0].device)
    
    # Initialize a mask tensor to indicate padding positions
    mask = torch.zeros(batch_size, max_length, feature_size, device=tensor_list[0].device)
    
    # Stack the original tensors along a new dimension
    for i, tensor in enumerate(tensor_list):
        length = tensor.size(0)
        padded_tensors[i, :length, :] = tensor
        mask[i, length:, :] = 1  # Mark the padded positions with 1
    
    return padded_tensors, mask
    
    return stacked_tensors
class EpisodicDataset_card(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, device):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.is_sim = None
        self.device = device
        self.gripper_dims = torch.tensor([0.063, 0.205, 0.0635], dtype=torch.float32, device=self.device)

        self.action_scale_low = [-0.1] * 7 + [-0.04] * 2 + [10.0] * 7 + [5.0] * 2 + [0.0] * 9
        self.action_scale_high = [0.1] * 7 + [0.04] * 2 + [200.0] * 4 + [100.0] * 5 + [2.0] * 9
        
        self.env_state_low = [-1.0] * (24 + 24 + 24 + 6 + 24 + 6) + self.action_scale_low
        self.env_state_high = [1.0] * (24 + 24 + 24 + 6 + 24 + 6) + self.action_scale_high 
        
        self.robot_state_low = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0] \
                                + [-2.175, -2.175, -2.175, -2.175, -2.61, -2.61, -2.61, -0.1, -0.1]
        self.robot_state_high = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04] \
                                + [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 0.1, 0.1] \

        self.action_scale_low = torch.tensor(self.action_scale_low, dtype=torch.float32, device=self.device)
        self.action_scale_high = torch.tensor(self.action_scale_high, dtype=torch.float32, device=self.device)
        self.env_state_low = torch.tensor(self.env_state_low, dtype=torch.float32, device=self.device)
        self.env_state_high = torch.tensor(self.env_state_high, dtype=torch.float32, device=self.device)
        self.robot_state_low = torch.tensor(self.robot_state_low, dtype=torch.float32, device=self.device)
        self.robot_state_high = torch.tensor(self.robot_state_high, dtype=torch.float32, device=self.device)
        
        num_data_to_use= 10
        data_path_list = glob(os.path.join(self.dataset_dir, "augment_*.pt"))[:num_data_to_use]

        self.load_data(data_path_list)
        self.__getitem__(0) # initialize self.is_sim
                
    def load_data(self, data_path_list):
        gripper_dims = torch.tensor([0.063, 0.205, 0.0635], dtype=torch.float32, device=self.device)

        terminations_idx = 0
        terminations_buf = []
        robot_state_buf = []
        env_state_buf = []
        action_buf = []
        for data_path in tqdm(data_path_list):
            data = torch.load(data_path)
            qO_hist = data["qO_hist"][1:-1]
            vO_hist = data["vO_hist"][1:-1]
            qR_hist = data["qR_hist"][1:-1]
            vR_hist = data["vR_hist"][1:-1]
            ee_hist = data["ee_hist"][1:-1]
            tip_hist = data["tip_hist"][1:-1]

            traj_len = qO_hist.shape[0]
            terminations_idx += traj_len
            terminations_buf.append(terminations_idx)

            obj_pose_to_keypoints_3d = PoseToKeyPoint(traj_len, device=self.device).gen_keypoints_3d
            ee_pose_to_keypoints_3d = PoseToKeyPoint(traj_len, device=self.device, size=gripper_dims).gen_keypoints_3d

            obj_keypoints_3d = obj_pose_to_keypoints_3d(qO_hist).reshape(traj_len, -1)
            G_keypoints_3d = obj_pose_to_keypoints_3d(data["qSG"][:7].repeat(traj_len, 1)).reshape(traj_len, -1)
            ee_keypoints = ee_pose_to_keypoints_3d(ee_hist).reshape(traj_len, -1)
            rel_ee_keypoints = ee_pose_to_keypoints_3d(get_relative_robot_pose(qO_hist, ee_hist)).reshape(traj_len, -1)
            rel_tip_position = get_relative_robot_tip_position(qO_hist, tip_hist.reshape(-1, 2, 3))
            prev_action = data["al_hist"][0:-2]
            
            robot_state = torch.cat([qR_hist, vR_hist], dim=-1)
            env_state = torch.cat([
                obj_keypoints_3d,   # 24
                G_keypoints_3d,     # 24
                ee_keypoints,       # 24
                tip_hist,           # 6
                rel_ee_keypoints,   # 24
                rel_tip_position,   # 6
                prev_action         # 27
            ], dim=-1)
            robot_state_buf.append(robot_state)
            env_state_buf.append(env_state)

            al_hist = data["al_hist"][1:-1]
            action = al_hist
            action_buf.append(action)

        self.terminations_buf = torch.tensor(terminations_buf, device=self.device).cpu()
        
        max_length_robot = max(tensor.size(0) for tensor in robot_state_buf)
        max_length_env = max(tensor.size(0) for tensor in env_state_buf)
        max_length_Action = max(tensor.size(0) for tensor in action_buf)
        if max_length_robot != max_length_env or max_length_robot != max_length_Action:
            raise ValueError("Robot, Env and Action lengths are not same")
        else:
            pass
        self.robot_state_buf, self.robot_state_pad = pad_and_stack_with_mask(robot_state_buf)
        self.env_state_buf, self.env_state_pad = pad_and_stack_with_mask(env_state_buf)
        self.action_buf, self.action_pad = pad_and_stack_with_mask(action_buf)

        print(f"Total number of plans: {len(robot_state_buf)}")
        print(f"max length of plan: {max_length_Action}")

    def __len__(self):
        return len(self.episode_ids)
    
    def normalize_env_state(self, env_state):
        return scale_transform(env_state, self.env_state_low, self.env_state_high)
        
    def normalize_robot_state(self, robot_state):
        return scale_transform(robot_state, self.robot_state_low, self.robot_state_high)
        
    def normalize_action(self, action):
        return scale_transform(action, self.action_scale_low, self.action_scale_high)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        action_pad = self.action_pad[index]
        non_zero_rows = torch.any(action_pad != 0, dim=1)
        episode_len = torch.sum(non_zero_rows).item()
        action_full = self.action_buf[index]
        original_action_shape = action_full.shape
        robot_state = self.robot_state_buf[index]
        env_state = self.env_state_buf[index]
        
        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        # get observation at start_ts only
        robot_state = robot_state[start_ts]
        env_state = env_state[start_ts]

        action = action_full[start_ts:episode_len]
        action_len = episode_len - start_ts

        padded_action = torch.zeros(original_action_shape, device=self.device).float()
        padded_action[:action_len] = action
        is_pad = torch.zeros(original_action_shape[0], device=self.device)
        is_pad[action_len:] = 1

        # construct observations
        is_pad = is_pad.bool()

        # normalize image and change dtype to float, N: normalize
        N_action_data = self.normalize_action(padded_action)
        N_action_data = N_action_data.clamp(-1.0, 1.0)
        
        N_env_state_data = self.normalize_env_state(env_state)
        N_robot_state_data = self.normalize_robot_state(robot_state)
        N_env_state_data = N_env_state_data.clamp(-5.0, 5.0)
        N_robot_state_data = N_robot_state_data.clamp(-5.0, 5.0)
        breakpoint()
        return N_env_state_data, N_robot_state_data, N_action_data, is_pad


class OfflineStateActionPairs(torch.utils.data.Dataset):
    def __init__(self, data_root, num_data_to_use=-1, device="cuda:0"):
        """_summary_

        Args:
            data_root (str): data 
        """
        super().__init__()
        self.data_root = data_root
        self.device = device

        data_path_list = glob(os.path.join(data_root, "augment_*.pt"))[:num_data_to_use]
        self.load_data(data_path_list)
        
        self.obs_dim = self.state_buf.shape[1]
        self.action_dim = self.action_buf.shape[1]

    def load_data(self, data_path_list):
        gripper_dims = torch.tensor([0.063, 0.205, 0.0635], dtype=torch.float32, device=self.device)

        terminations_idx = 0
        terminations_buf = []
        state_buf = []
        action_buf = []
        for data_path in tqdm(data_path_list):
            data = torch.load(data_path)
            qO_hist = data["qO_hist"][1:-1]
            vO_hist = data["vO_hist"][1:-1]
            qR_hist = data["qR_hist"][1:-1]
            vR_hist = data["vR_hist"][1:-1]
            ee_hist = data["ee_hist"][1:-1]
            tip_hist = data["tip_hist"][1:-1]

            traj_len = qO_hist.shape[0]
            terminations_idx += traj_len
            terminations_buf.append(terminations_idx)

            obj_pose_to_keypoints_3d = PoseToKeyPoint(traj_len, device=self.device).gen_keypoints_3d
            ee_pose_to_keypoints_3d = PoseToKeyPoint(traj_len, device=self.device, size=gripper_dims).gen_keypoints_3d

            obj_keypoints_3d = obj_pose_to_keypoints_3d(qO_hist).reshape(traj_len, -1)
            G_keypoints_3d = obj_pose_to_keypoints_3d(data["qSG"][:7].repeat(traj_len, 1)).reshape(traj_len, -1)
            ee_keypoints = ee_pose_to_keypoints_3d(ee_hist).reshape(traj_len, -1)
            rel_ee_keypoints = ee_pose_to_keypoints_3d(get_relative_robot_pose(qO_hist, ee_hist)).reshape(traj_len, -1)
            rel_tip_position = get_relative_robot_tip_position(qO_hist, tip_hist.reshape(-1, 2, 3))
            prev_action = data["al_hist"][0:-2]
            state = torch.cat([
                qR_hist,            # 9
                vR_hist,            # 9
                obj_keypoints_3d,   # 24
                G_keypoints_3d,     # 24
                ee_keypoints,       # 24
                tip_hist,           # 6
                rel_ee_keypoints,   # 24
                rel_tip_position,   # 6
                prev_action         # 27
            ], dim=-1)
            state_buf.append(state)

            al_hist = data["al_hist"][1:-1]
            action = al_hist
            action_buf.append(action)

        self.terminations_buf = torch.tensor(terminations_buf, device=self.device).cpu()
        self.state_buf = torch.cat(state_buf).cpu()
        self.action_buf = torch.cat(action_buf).cpu()

        print(f"Total number of plans: {len(state_buf)}")
        print(f"Total number of states: {self.state_buf.shape[0]}")

    def get_eval_dataset(self, fraction=0.2):
        self.max_train_idx = int(np.ceil((1 - fraction) * len(self.state_buf)))

        eval_dataset = deepcopy(self)
        eval_dataset.state_buf = eval_dataset.state_buf[self.max_train_idx:]
        eval_dataset.action_buf = eval_dataset.action_buf[self.max_train_idx:]

        self.state_buf = self.state_buf[:self.max_train_idx]
        self.action_buf = self.action_buf[:self.max_train_idx]
        
        return eval_dataset

    def __len__(self):
        return len(self.state_buf)
    
    def __getitem__(self, idx):
        return self.state_buf[idx], self.action_buf[idx]
    
    def save_as_zarr(self, path=None):
        import zarr
        if path is None:
            path = self.data_root + ".zarr"
        
        group = zarr.group(path)
        
        group.create_group("meta")
        group["meta"].create_dataset("episode_ends", data=self.terminations_buf.cpu().numpy())

        group.create_group("data")
        group["data"].create_dataset("state", data=self.state_buf.cpu().numpy())
        group["data"].create_dataset("action", data=self.action_buf.cpu().numpy())

    def plot_action(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(40, 15))
        for i in range(27):
            plt.subplot(3,9,i+1)
            plt.hist(self.action_buf[:,i].cpu(), bins=50)
        plt.show()

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, batch_size_train, batch_size_val, device):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    # norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    # train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    # val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataset = EpisodicDataset_card(train_indices, dataset_dir, device=device)
    val_dataset = EpisodicDataset_card(val_indices, dataset_dir, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
