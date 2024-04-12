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
        self._ball_position = [0.0, 2.0, 0.0] 
        self._goal_position = [0.0, 6.0, 0.0]

        # values used for defining RL buffers
        self._num_observations = 3
        self._num_actions = 6
        self._device = "cpu"
        self.num_envs = 1

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
        # retrieve file path for the Cartpole USD file
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur3/ur3.usd"
        # add the Cartpole USD to our stage
        create_prim(prim_path="/World/Robot", prim_type="Xform", position=self._robot_arm_pos)
        add_reference_to_stage(usd_path, "/World/Robot")
        create_prim(prim_path="/World/Ball", prim_type="Sphere", attributes={"radius": 0.1}, position=self._ball_position)
        create_prim(prim_path="/World/Goal", prim_type="Sphere", attributes={"radius": 0.05}, position=self._goal_position)
        # create an ArticulationView wrapper for our cartpole - this can be extended towards accessing multiple cartpoles
        self._robot = ArticulationView(prim_paths_expr="/World/Robot*", name="robot_view")
        self.tool_view = RigidPrimView(prim_paths_expr="/World/Robot/tool0", name="tool_view", reset_xform_properties=False)
        # add Cartpole ArticulationView and ground plane to the Scene
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

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        
        # Reset ball and goal positions
        # TODO random goal positionining here
        self.resets[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        actions = torch.tensor(actions)

        forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[0]

        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)
    
    def get_observations(self):
        tool_pos, tool_rot = self.tool_view.get_world_poses(clone=False) 
        self.obs = tool_pos
        return self.obs
    
    def is_done(self) -> None:
        cart_pos = self.obs[:, 0]
        pole_pos = self.obs[:, 2]

        # reset the robot if cart has reached reset_dist or pole is too far from upright
        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        self.resets = resets

        return resets.item()