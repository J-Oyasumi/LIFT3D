import gymnasium
from gymnasium import Wrapper
import torch
import tqdm
import numpy as np 
from termcolor import colored
import mani_skill.envs
from lift3d.envs.evaluator import Evaluator
from diffusion_policy_3d.common.utils import downsample_with_fps


TASK_BOUDNS = {
    'GraspCup-v1': [-0.7, -1, 0.01, 1, 1, 100],
    'PickCube-v1': [-0.7, -1, 0.01, 1, 1, 100],
    'default': [-0.7, -1, 0.01, 1, 1, 100],
}


TASK_DESCRIPTION = {
    "PickCube-v1":"grasp a red cube and move it to a target goal position."
}

MAX_EPISODE_LENGTH = {
    "PickCube-v1":50,
}

class ManiSkillEnv(gymnasium.Env):
    def __init__(
        self,
        task_name,
        use_point_crop=True,
        num_points=1024,
    ):
        super(ManiSkillEnv, self).__init__()
        
        self.task_name = task_name

        self.env = gymnasium.make(
            task_name,
            num_envs=1,
            obs_mode="pointcloud",
            control_mode="pd_ee_delta_pose"    
        )
        self.max_episode_length = MAX_EPISODE_LENGTH[task_name]
        self.text = TASK_DESCRIPTION[task_name]  
        print(f'task description:{self.text}')
        
        self.use_point_crop = use_point_crop
        self.num_points = num_points
        
        self.step_count = 0
    
    def get_robot_state(self):
        obs = self.env.get_wrapper_attr('get_obs')()
        qpos = obs['agent']['qpos'][...][0]
        qvel = obs['agent']['qvel'][...][0]
        tcp_pose = obs['extra']['tcp_pose'][...][0]
        robot_state = np.concatenate((qpos, qvel, tcp_pose))

        return robot_state
    
    def get_point_cloud(self):
        obs = self.env.get_wrapper_attr('get_obs')()
        xyzw = obs["pointcloud"]['xyzw'][...][0]
        rgb = obs["pointcloud"]['rgb'][...][0]
        filtered_xyzw = xyzw[xyzw[:, -1] == 1]
        filtered_xyz = filtered_xyzw[...,:3]
        filtered_rgb = rgb[xyzw[:, -1] == 1]
        filtered_xyz = filtered_xyz.cpu().numpy()
        filtered_rgb = filtered_rgb.cpu().numpy()
        obs_point_cloud = np.concatenate((filtered_xyz, filtered_rgb), axis=1)
            
        if self.task_name in TASK_BOUDNS:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS[self.task_name]
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS['default']
        min_bound = [x_min, y_min, z_min]
        max_bound = [x_max, y_max, z_max]        

        if self.use_point_crop:
            if min_bound is not None:
                mask = np.all(obs_point_cloud[:, :3] > min_bound, axis=1)
                obs_point_cloud = obs_point_cloud[mask]
            if max_bound is not None:
                mask = np.all(obs_point_cloud[:, :3] < max_bound, axis=1)
                obs_point_cloud = obs_point_cloud[mask]
            
        if obs_point_cloud.shape[0] > self.num_points:
            obs_point_cloud = downsample_with_fps(obs_point_cloud, self.num_points)
            
        
        return obs_point_cloud
    
    def get_rgb(self):
        obs = self.env.get_wrapper_attr('get_obs')()
        image = obs['sensor_data']['base_camera']['rgb'][0]
        image = image.cpu().numpy()
        return image
    
    def get_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        # raw_state = self.env.get_obs() 
        point_cloud = self.get_point_cloud()
        obs_dict = {
            "image": obs_pixels,
            "robot_state": robot_state,
            "raw_state": torch.zeros((0,)),
            "point_cloud": point_cloud,
        }
        return obs_dict
    
    def reset(self):
        self.env.reset()
        self.step_count = 0
        return self.get_obs()
    
    def step(self, action):
        _, _, terminated, _, _ = self.env.step(action)
        obs = self.get_obs()
        self.step_count += 1
        done = terminated or self.step_count >= self.max_episode_length
        success = self.env.get_wrapper_attr('evaluate')()['success'].item()
        print(f'step: {self.step_count}, success:{success}')
        return obs, done, success 
                
class ManiSkillEvaluator():
    def __init__(
        self,
        task_name,
        use_point_crop=True,
        num_points=1024,
    ):
        self.env = ManiSkillEnv(
            task_name=task_name,
            use_point_crop=use_point_crop,
            num_points=num_points,
        )
        

    def evaluate(self, num_episodes, policy, verbose: bool = False):
        # task_name = Wrapper.get_wrapper_attr(self.env, "task_name")
        task_name = self.env.task_name
        if verbose:
            success_list, rewards_list = [], []
            video_steps_list = []
        else:
            total_success, total_rewards = 0, 0

        for i in tqdm.tqdm(
            range(num_episodes),
            desc=f'Evaluating in ManiSkill <{colored(task_name, "red")}>',
        ):
            obs_dict = self.env.reset()
            done = False
            success = False
    
            while not done and not success:
                obs_img = obs_dict["image"]
                obs_point_cloud = obs_dict["point_cloud"]
                obs_robot_state = obs_dict["robot_state"]
                device = next(policy.parameters()).device
                obs_img_tensor = (
                    torch.from_numpy(obs_img).float().unsqueeze(0).to(device)
                )
                obs_point_cloud_tensor = (
                    torch.from_numpy(obs_point_cloud).float().unsqueeze(0).to(device)
                )
                obs_robot_state_tensor = (
                    torch.from_numpy(obs_robot_state).float().unsqueeze(0).to(device)
                )
                obs_img_tensor = obs_img_tensor.permute(0, 3, 1, 2)
                batch_size = obs_img_tensor.shape[0]
                input_data = {
                    "images": obs_img_tensor,
                    "point_clouds": obs_point_cloud_tensor,
                    "robot_states": obs_robot_state_tensor,
                    "texts": [self.env.text] * batch_size,
                }
                with torch.no_grad():
                    action = policy(**input_data)
                action = action.to("cpu").detach().numpy().squeeze()
                obs_dict, done, success = self.env.step(action)
                
            if success:
                print("success!")
            else:
                print("fail!")
                
            if verbose:
                video_steps_list.append(self.env.get_frames().transpose(0, 3, 1, 2))
                success_list.append(success)
            else:
                total_success += success

        if verbose:
            return_value = (
                sum(success_list) / num_episodes,
            )
            self.success_list = success_list
            # self.rewards_list = rewards_list
            self.video_steps_list = video_steps_list
        else:
            avg_success = total_success / num_episodes
            # avg_rewards = total_rewards / num_episodes
            return_value = avg_success 

        return return_value

    def callback_verbose(self, wandb_logger):
        import plotly.express as px
        import plotly.graph_objects as go
        import wandb

        fig1 = go.Figure(
            data=[
                go.Bar(
                    x=["Success", "Failure"],
                    y=[
                        sum(self.success_list),
                        len(self.success_list) - sum(self.success_list),
                    ],
                )
            ]
        )
        wandb_logger.log({"Charts/success_failure": fig1})

