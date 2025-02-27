from RoboTwin.envs.block_hammer_beat import block_hammer_beat
import gymnasium
from gymnasium import Wrapper
import tqdm
import torch
from termcolor import colored
import numpy as np
from lift3d.envs.evaluator import Evaluator
from lift3d.helpers.gymnasium import VideoWrapper

TASK_DESCRIPTION = {
    "block_hammer_beat":'If the block is closerto the left robotic arm, it uses the left arm to pick up the hammer and strike the\
                block; otherwise, it does the opposite.'
}

kwargs ={
    'task_name': 'block_hammer_beat',
    'render_freq':0,
    'pcd_crop':True,
    'pcd_down_sample_num':1024,
    'data_type':{
        'rgb':True,
        'observer':True,
        'depth':False,
        'point_cloud':True,
        'end_pose':True,
        'qpos':True,
        'mesh_segmentation':False,
        'actor_segmentation':False,
    }
}

class RoboTwinEnv(gymnasium.Env):
    def __init__(
        self,
        task_name,
        use_point_crop=True,
        num_points=1024,
        # point_cloud_camera_names=["corner"],
    ):
        super(RoboTwinEnv, self).__init__()

        self.task_name = task_name
        if task_name == "block_hammer_beat":
            self.env = block_hammer_beat()
            self.max_episode_length = 400
            self.env.setup_demo(**kwargs)
        else:
            raise ValueError(f"Invalid task name: {task_name}")
        
        # self.camera_name = camera_name
        # self.point_cloud_camera_names = point_cloud_camera_names
        # self.image_w = image_w
        # self.image_h = image_h
        self.use_point_crop = use_point_crop # these two is not used
        self.num_points = num_points
        
        self.text = TASK_DESCRIPTION[task_name]
                
        self.step_count = 0
        
    def get_robot_state(self):
        obs = self.env.get_obs()
        joint_action_shape = obs["joint_action"].shape     
        return obs["joint_action"]
    
    def get_point_cloud(self):
        obs = self.env.get_obs()
        pointcloud_shape = obs["pointcloud"].shape
        return obs["pointcloud"]

    def get_rgb(self):
        obs = self.env.get_obs()
        rgb_shape = obs["observation"]["observer_camera"]["rgb"].shape
        return obs["observation"]["observer_camera"]["rgb"]
        
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
        self.env = block_hammer_beat()
        self.env.setup_demo(**kwargs)
        self.step_count = 0
        return self.get_obs()
    
    def step(self, action):
        obs = self.get_obs()
        self.step_count += 1
        # step in the environment(according to apply_policy_demo)
        left_arm_actions, left_gripper = action[:6], action[6] # 0-5 left joint action, 6 left gripper action
        right_arm_actions, right_gripper = action[7:13], action[13] # 7-12 right joint action, 13 right gripper action
        left_current_qpos, right_current_qpos = obs['robot_state'][:6], obs['robot_state'][7:13]
        left_path = np.vstack((left_current_qpos, left_arm_actions))
        right_path = np.vstack((right_current_qpos, right_arm_actions))
        topp_left_flag, topp_right_flag = True, True
        try:
            times, left_pos, left_vel, acc, duration = self.left_planner.TOPP(left_path, 1/250, verbose=True)
            left_result = dict()
            left_result['position'], left_result['velocity'] = left_pos, left_vel
            left_n_step = left_result["position"].shape[0]
            left_gripper = np.linspace(left_gripper[0], left_gripper[-1], left_n_step)
        except:
            topp_left_flag = False
            left_n_step = 1       
        try:
            times, right_pos, right_vel, acc, duration = self.right_planner.TOPP(right_path, 1/250, verbose=True)            
            right_result = dict()
            right_result['position'], right_result['velocity'] = right_pos, right_vel
            right_n_step = right_result["position"].shape[0]
            right_gripper = np.linspace(right_gripper[0], right_gripper[-1], right_n_step)
        except:
            topp_right_flag = False
            right_n_step = 1
            
        if right_n_step == 0:
            topp_right_flag = False
            right_n_step = 1
        
            now_left_id = 0 if topp_left_flag else 1e9
            now_right_id = 0 if topp_right_flag else 1e9
            
            while now_left_id < left_n_step or now_right_id < right_n_step:
                qf = self.robot.compute_passive_force(
                    gravity=True, coriolis_and_centrifugal=True
                )
                self.robot.set_qf(qf)
                if topp_left_flag and now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step:
                    for j in range(len(self.left_arm_joint_id)):
                        left_j = self.left_arm_joint_id[j]
                        self.active_joints[left_j].set_drive_target(left_result["position"][now_left_id][j])
                        self.active_joints[left_j].set_drive_velocity_target(left_result["velocity"][now_left_id][j])
                    if not self.fix_gripper:
                        for joint in self.active_joints[34:36]:
                            joint.set_drive_target(left_gripper[now_left_id])
                            joint.set_drive_velocity_target(0.05)
                            self.left_gripper_val = left_gripper[now_left_id]

                    now_left_id +=1
                    
                if topp_right_flag and now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step:
                    for j in range(len(self.right_arm_joint_id)):
                        right_j = self.right_arm_joint_id[j]
                        self.active_joints[right_j].set_drive_target(right_result["position"][now_right_id][j])
                        self.active_joints[right_j].set_drive_velocity_target(right_result["velocity"][now_right_id][j])
                    if not self.fix_gripper:
                        for joint in self.active_joints[36:38]:
                            joint.set_drive_target(right_gripper[now_right_id])
                            joint.set_drive_velocity_target(0.05)
                            self.right_gripper_val = right_gripper[now_right_id]

                    now_right_id +=1    
            
        self.env.scene.step()
        obs = self.get_obs()
        done = self.step_count >= self.max_episode_length 
        success = self.env.check_success()
        
        return obs, done, success
    
        

class RoboTwinEvaluator(Evaluator):
    def __init__(
        self,
        task_name,
        # image_w=320,
        # image_h=180,
        # camera_name="corner",
        use_point_crop=True,
        num_points=1024,
        # point_cloud_camera_names=["corner"],
    ):
        self.env = RoboTwinEnv(
            task_name=task_name,
            # max_episode_length=max_episode_length,
            # image_w=image_w,
            # image_h=image_h,
            # camera_name=camera_name,
            use_point_crop=use_point_crop,
            num_points=num_points,
            # point_cloud_camera_names=point_cloud_camera_names,
        )
        # self.env = VideoWrapper(self.env)

    def evaluate(self, num_episodes, policy, verbose: bool = False):
        task_name = Wrapper.get_wrapper_attr(self.env, "task_name")

        if verbose:
            success_list, rewards_list = [], []
            video_steps_list = []
        else:
            total_success, total_rewards = 0, 0

        for i in tqdm.tqdm(
            range(num_episodes),
            desc=f'Evaluating in RoboTwin <{colored(task_name, "red")}>',
        ):
            # obs_dict = Wrapper.get_wrapper_attr(self.env, "get_obs")()
            obs_dict = self.env.reset()
            # truncated = terminated = False
            # rewards = 0
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
                # rewards_list.append(rewards)
            else:
                total_success += success
                # total_rewards += rewards

        if verbose:
            return_value = (
                sum(success_list) / num_episodes,
                # sum(rewards_list) / num_episodes,
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

        # for i, (success, video_steps) in enumerate(
        #     zip(self.success_list, self.video_steps_list)
        # ):
        #     if success:
        #         wandb_logger.log(
        #             {
        #                 f"validation/video_steps_success_{i}": wandb.Video(
        #                     video_steps, fps=30
        #                 ),
        #             }
        #         )
        #     else:
        #         wandb_logger.log(
        #             {
        #                 f"validation/video_steps_failure_{i}": wandb.Video(
        #                     video_steps, fps=30
        #                 ),
        #             }
        #         )