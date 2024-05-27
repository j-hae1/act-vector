from torch_jit_utils import *
from scipy.spatial.transform import Rotation as R
from typing import Tuple
import numpy as np
import torch

@torch.jit.script
def gen_keypoints(pose: torch.Tensor, num_keypoints: int = 8, size: Tuple[float, float, float] = (0.065, 0.065, 0.065)) -> torch.Tensor:
    num_envs = pose.shape[0]
    keypoints_buf = torch.ones((num_envs, num_keypoints, 3), dtype=torch.float32, device=pose.device)
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * keypoints_buf[:, i, :]
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose)
    return keypoints_buf

@torch.jit.script
def compute_projected_points(T_matrix: torch.Tensor, keypoints: torch.Tensor, camera_matrix: torch.Tensor, device: str, num_points: int = 8) -> torch.Tensor:
    num_envs = keypoints.shape[0]
    p_CO = torch.matmul(T_matrix, torch.cat([keypoints, torch.ones((num_envs, num_points, 1), device=device)], -1).transpose(1, 2))
    image_coordinates = torch.matmul(camera_matrix, p_CO).transpose(1, 2)
    mapped_coordinates = image_coordinates[:, :, :2] / (image_coordinates[:, :, 2].unsqueeze(-1))
    return mapped_coordinates


class PoseToKeyPoint:
    def __init__(self, num_envs, device, num_keypoints=8, size=(0.05, 0.07, 0.005)):
        self.device = device
        _camera_position = torch.tensor([0.96, 0, 0.86], device=self.device).unsqueeze(-1)
        _camera_angle = 43.0
        rotation_matrix = torch.tensor((R.from_rotvec(np.array([0., 1., 0.]) * np.radians(-90 - _camera_angle)) * R.from_rotvec(np.array([0., 0., 1.,]) * np.radians(90))).inv().as_matrix(), dtype=torch.float).to(self.device)
        self.translation_from_camera_to_object = torch.zeros((3, 4), device=self.device)
        self.translation_from_camera_to_object[:3, :3] = rotation_matrix
        self.translation_from_camera_to_object[:3, 3] = -rotation_matrix.mm(_camera_position)[:, 0]
        self.camera_matrix = self.compute_camera_intrinsics_matrix(320, 240, 55.368, self.device)

        self.num_envs = num_envs
        self.num_keypoints = num_keypoints
        self.keypoints_buf = torch.ones((num_envs, num_keypoints, 3), dtype=torch.float32, device=device)
        for i in range(num_keypoints):
            # which dimensions to negate
            n = [((i >> k) & 1) == 0 for k in range(3)]
            corner_loc = [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
            self.keypoints_buf[:, i, :] = torch.tensor(corner_loc, dtype=torch.float32, device=device)

    def compute_camera_intrinsics_matrix(self, image_width, image_heigth, horizontal_fov, device) -> torch.Tensor:
        vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
        horizontal_fov *= np.pi / 180
        f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
        f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)
        K = torch.tensor([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        return K
    
    def gen_keypoints_3d(self, pose: torch.Tensor) -> torch.Tensor:
        B = pose.shape[0]
        if self.num_envs == B:
            key_points_flatten = local_to_world_space(self.keypoints_buf.reshape(-1,3), 
                                                      pose.unsqueeze(1).repeat(1,self.num_keypoints,1).reshape(-1,7))
        else:
            key_points_flatten = local_to_world_space(self.keypoints_buf[:B].reshape(-1,3), 
                                                      pose.unsqueeze(1).repeat(1,self.num_keypoints,1).reshape(-1,7))
        
        key_points = key_points_flatten.reshape(-1, self.num_keypoints, 3)
        
        return key_points
    
    def gen_keypoints_2d(self, pose: torch.Tensor) -> torch.Tensor:
        key_points = self.gen_keypoints_3d(pose)
        return compute_projected_points(self.translation_from_camera_to_object, key_points, self.camera_matrix, self.device)

def get_relative_robot_pose(absolute_object_pose: torch.Tensor, absolute_robot_pose: torch.Tensor):
    """_summary_

    Args:
        absolute_object_pose (torch.Tensor): shape with B, 7(x, y, z, qx, qy, qz, qw)
        absolute_robot_pose (torch.Tensor): shape with B, 7(x, y, z, qx, qy, qz, qw)
    """
    inv_object_quat, inv_object_pos = tf_inverse(absolute_object_pose[:, 3:7], absolute_object_pose[:, :3])
    rel_robot_quat, rel_robot_pos = tf_combine(inv_object_quat, inv_object_pos, absolute_robot_pose[:, 3:7], absolute_robot_pose[:, :3])

    return torch.cat([rel_robot_pos, rel_robot_quat], dim=-1)
    
def get_relative_robot_tip_position(absolute_object_pose: torch.Tensor, absolute_tip_position: torch.Tensor):
    """_summary_

    Args:
        absolute_object_pose (torch.Tensor): shape with B, 7(x, y, z, qx, qy, qz, qw)
        absolute_tip_position (torch.Tensor): shape with B, 2, 3 (left and right tip with xyz)
    """
    obj_quat = absolute_object_pose[:, 3:7]
    obj_pos = absolute_object_pose[:, :3]
    inv_obj_quat = quat_conjugate(obj_quat)

    # Subtract the object's translation from the absolute positions
    left_tip = absolute_tip_position[:, 0]
    right_tip = absolute_tip_position[:, 1]  
    rel_left_positions = left_tip - obj_pos
    rel_right_positions = right_tip - obj_pos

    # Apply the inverse quaternion to the relative positions
    relative_left_tip_positions = quat_apply(inv_obj_quat, rel_left_positions)
    relative_right_tip_positions = quat_apply(inv_obj_quat, rel_right_positions)

    relative_tip_positions = torch.cat([relative_left_tip_positions, relative_right_tip_positions], dim=1)

    return relative_tip_positions