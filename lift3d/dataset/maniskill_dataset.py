import torch
import zarr
from torch.utils.data import Dataset

# point_cloud shape: (1024, 6) dtype: float32
# robot_state shape: (25,)  dtype: float32
# joint positions[7+2] velocities[7+2] end-effector pose [7]
# action shape: (7,)  dtype: float32
# images shape: (3, 128, 128)  dtype: uint8

TASK_DESCRIPTION = {
    "PickCube-v1":"A simple task where the objective is to grasp a red cube and move it to a target goal position."
}

class ManiSkillDataset(Dataset):
    """
    Dataset for ManiSkill Benchmark.
    """
    
    SPLIT_SIZE = {"train": 90, "validation": 10, "custom": None}
    
    def __init__(self, data_dir, task_name, split: str = None, custom_split_size: int = None):
        zarr_root = zarr.open_group(data_dir, mode="r")
        self.task_name = task_name
        self._episode_ends = zarr_root["meta"]["episode_ends"][:]
        
        if split not in self.SPLIT_SIZE:
            raise ValueError(f"Invalid split: {split}")
        
        if split == "custom" and custom_split_size is None:
            raise ValueError(f"custom_split_size must be provided for split: {split}")
        
        begin_index, end_index = (
            (0, self._episode_ends[self.SPLIT_SIZE["train"] - 1])
            if split == "train"
            else (
                (
                    self._episode_ends[self.SPLIT_SIZE["train"] - 1],
                    self._episode_ends[
                        self.SPLIT_SIZE["train"] + self.SPLIT_SIZE["validation"] - 1
                    ],
                )
                if split == "validation"
                else (0, self._episode_ends[custom_split_size - 1])
            )
        )
        self._images = zarr_root["data"]["image"][begin_index:end_index].transpose(
            0, 3, 1, 2
        )        
        self._point_clouds = zarr_root["data"]["point_cloud"][begin_index:end_index]
        self._robot_states = zarr_root["data"]["state"][begin_index:end_index]
        self._actions = zarr_root["data"]["action"][begin_index:end_index]
        
        self._dataset_size = len(self._actions)
        
    def __getitem__(self, idx):
        point_cloud = torch.from_numpy(self._point_clouds[idx]).float()
        robot_state = torch.from_numpy(self._robot_states[idx]).float()
        action = torch.from_numpy(self._actions[idx]).float()
        image = torch.from_numpy(self._images[idx]).float()
        text = TASK_DESCRIPTION[self.task_name]
        return image, point_cloud, robot_state, torch.zeros((0,)), action, text 
    
    def __len__(self):
        return self._dataset_size
    
    def print_info(self):
        print("ManiSkill Dataset Info:")
        print(f'point_cloud shape: {self._point_clouds.shape}, dtype: {self._point_clouds.dtype}')
        print(f'robot_state shape: {self._robot_states.shape}, dtype: {self._robot_states.dtype}')
        print(f'action shape: {self._actions.shape}, dtype: {self._actions.dtype}')
        print(f'image shape: {self._images.shape}, dtype: {self._images.dtype}')
        print(f'episode_ends: {self._episode_ends}')


if __name__ == "__main__":
    data_dir = "data/maniskill/maniskill_PickCube-v1_expert.zarr"
    dataset = ManiSkillDataset(data_dir, task_name="PickCube-v1", split="custom", custom_split_size=90)
    print(f"Dataset size: {len(dataset)}")
    dataset.print_info()
