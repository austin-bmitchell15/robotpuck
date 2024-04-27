from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims import RigidPrimView

from omni.isaac.core.utils.viewports import set_camera_view

from gymnasium import spaces
import numpy as np
import torch
import math


class RobotPuckTask(BaseTask):
    def __init__(
        self,
        name,
        offset=None
    ) -> None:

        # task-specific parameters
        self._robot_arm_pos = [0.0, 0.0, 0.0]
        # TODO randomize these
        #self._ball_position = torch.rand(3) * 0.25
        self._ball_position = torch.tensor([0.4, 0.0, 0.4])
        self._goal_position = [0.0, 6.0, 0.0]
        
        #self._max_push_effort = torch.Tensor([3360.0, 3360.0, 1680.0, 720.0, 720.0, 720.0,])
        self.max_velocity_command = 0.5

        # Normalize observations
        # Elbow joint appears to have limit of math.pi, but probably doesn't matter
        self.norm_dof_pos = math.pi * 2.0
        #Obs may exceed max_vel_command slightly, but assuming it's fine
        #Obviously will not work in force control mode
        self.norm_dof_vel = self.max_velocity_command

        self.prev_distance = -1.0

        # values used for defining RL buffers
        self._num_observations = 22
        self._num_actions = 6
        self._device = "cpu"
        self.num_envs = 1

        self.tool_obs_slice = slice(0, 3)
        self.dof_pos_obs_slice = slice(3, 9)
        self.dof_vel_obs_slice = slice(9, 15)
        self.ball_pos_obs_slice = slice(15, 18)
        self.target_dist_obs_slice = slice(18, 21)
        self.time_obs_slice = slice(21, 22)

        # Vars for fixed episodes
        self.episode_length = 250
        self.t = torch.zeros(self.num_envs, dtype = torch.int)

        # a few class buffers to store RL-related states
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(
            np.ones(self._num_actions, dtype=np.float32) * -1.0, np.ones(self._num_actions, dtype=np.float32) * 1.0
        )
        self.observation_space = spaces.Box(
            np.ones(self._num_observations, dtype=np.float32) * -np.Inf,
            np.ones(self._num_observations, dtype=np.float32) * np.Inf,
        )


        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def set_up_scene(self, scene) -> None:
        # retrieve file path for the robot USD file
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur3/ur3.usd"
        # add the robot USD to our stage
        create_prim(prim_path="/World/Robot", prim_type="Xform", position=self._robot_arm_pos)
        add_reference_to_stage(usd_path, "/World/Robot")
        create_prim(prim_path="/World/Ball", prim_type="Sphere", attributes={"radius": 0.1}, position=self._ball_position)
        create_prim(prim_path="/World/Goal", prim_type="Sphere", attributes={"radius": 0.05}, position=self._goal_position)

        self._robot = ArticulationView(prim_paths_expr="/World/Robot*", name="robot_view")
        self.tool_view = RigidPrimView(prim_paths_expr="/World/Robot/tool0", name="tool_view", reset_xform_properties=False)

        # add ArticulationView and ground plane to the Scene
        scene.add(self.tool_view) 
        scene.add(self._robot)
        scene.add_default_ground_plane()

        # set default camera viewport position and target
        self.set_initial_camera_params()

    def set_initial_camera_params(self, camera_position=[5, 5, 10], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def post_reset(self):
        # TODO add random goal positioning here
        indices = torch.arange(self._robot.count, dtype=torch.int64, device=self._device)
        self.reset(indices)
        self._robot.switch_control_mode("velocity")
        print(self._robot.dof_names)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        
        num_resets = len(env_ids)
        
        # Reset ball and goal positions
        # TODO random goal positionining here

        dof_pos = torch.zeros((num_resets, self._robot.num_dof), device=self._device)
        dof_vel = torch.zeros((num_resets, self._robot.num_dof), device=self._device)

        indices = env_ids.to(dtype=torch.int32)
        self._robot.set_joint_positions(dof_pos, indices=indices)
        self._robot.set_joint_velocities(dof_vel, indices=indices)

        print("Resetting")
        self.t[env_ids] = 0

        self.resets[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        actions = torch.tensor(actions)

        velocities = torch.zeros((self._robot.count, self._robot.num_dof), dtype=torch.float32, device=self._device)
        velocities[:, :] = self.max_velocity_command * actions[0]

        indices = torch.arange(self._robot.count, dtype=torch.int32, device=self._device)
        self._robot.set_joint_velocities(velocities, indices = indices)

        self.t += 1
    
    def get_observations(self):
        tool_pos, tool_rot = self.tool_view.get_world_poses(clone=False)

        dof_pos = self._robot.get_joint_positions()
        dof_vel = self._robot.get_joint_velocities()

        #if torch.max(torch.abs(dof_pos / self.norm_dof_pos)) > 1.01:
            #print("Warning: dof_pos larger than expected.", dof_pos)
        #if torch.max(torch.abs(dof_vel / self.norm_dof_vel)) > 1.01:
            #print("Warning: dof_vel larger than expected.", dof_vel)

        self.obs[:, self.tool_obs_slice] = tool_pos
        self.obs[:, self.dof_pos_obs_slice] = dof_pos / self.norm_dof_pos
        self.obs[:, self.dof_vel_obs_slice] = dof_vel / self.norm_dof_vel
        self.obs[:, self.target_dist_obs_slice] = tool_pos - self._ball_position
        self.obs[:, self.time_obs_slice] = self.t / float(self.episode_length)
        return self.obs

    def calculate_metrics(self) -> None:
        tool_pos = self.obs[:, self.tool_obs_slice]
        dof_vel = self.obs[:, self.dof_vel_obs_slice]
        target = self._ball_position
        curr_distance = torch.sqrt(torch.square(tool_pos[:, 0]  - target[0]) + torch.square(tool_pos[:, 1]  - target[1]) + torch.square(tool_pos[:, 2]  - target[2]))
        # if curr_distance > 0.7:
        #     reward -= 2.0

        # if self.prev_distance < 0:
        #     pass
        # else:
        #     mult = 1.5 if curr_distance > self.prev_distance else 1.0
        #     reward += mult * (self.prev_distance - curr_distance)

        # if curr_distance < 0.05:
        #     reward += 0.5
        reward = 1.0 - 10.0 * curr_distance*curr_distance - 0.1 * torch.sum(torch.abs(dof_vel / self.max_velocity_command)) / 6
        reward = torch.where(curr_distance > 0.4, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(curr_distance < 0.05, torch.ones_like(reward) * 2.0, reward)


        self.prev_distance = curr_distance
        return reward.item()

    def is_done(self) -> None:
        tool_pos = self.obs[:, self.tool_obs_slice]
        target = self._ball_position

        #curr_distance = torch.sqrt(torch.square(tool_pos[:, 0]  - target[0]) + torch.square(tool_pos[:, 1]  - target[1]) + torch.square(tool_pos[:, 2]  - target[2]))
        
        resets = torch.where(self.t >= self.episode_length, 1, 0)
        self.resets = resets

        return resets.item()
