import copy
import os

import gymnasium as gym
import numpy as np
from simulator.common.action import action_factory
from simulator.common.graphics import EnvViewer
from simulator.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from simulator.road.road import Road, RoadNetwork
from simulator.vehicle.kinematics import Vehicle

from simulator.intersection.config import *


class AbstractEnv(gym.Env):

    """
    A generic environment for various tasks involving a vehicle driving on a road.

    The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
    speed. The action space is fixed, but the observation space and reward function must be defined in the
    environment implementations.
    """

    # PERCEPTION_DISTANCE = 5.0 * Vehicle.MAX_SPEED
    """The maximum distance of any vehicle present in the observation [m]"""

    def __init__(self, config: dict = None, render=False) -> None:
        super().__init__()

        # Configuration
        self.config = self.default_config()
        self.configure(config)

        # Scene
        self.road = None
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self._record_video_wrapper = None
        
        self.render = render
        self.enable_auto_render = False

        self.reset()

    @property
    def vehicle(self) -> Vehicle:
        """First (default) controlled vehicle."""
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        """Set a unique controlled vehicle."""
        self.controlled_vehicles = [vehicle]

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "ContinuousAction"
            },
            "simulation_frequency": 10,  # [Hz]
            "policy_frequency": 10,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 300,  # [px]
            "screen_height": 300,  # [px]
            "centering_position": [0, 0],
            "scaling": 4,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            "real_time_rendering": False,
        }

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    def update_metadata(self, video_real_time_ratio=2):
        frames_freq = self.config["simulation_frequency"] \
            if self._record_video_wrapper else self.config["policy_frequency"]
        self.metadata['render_fps'] = video_real_time_ratio * frames_freq

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        # self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        # self.action_type = ContinuousAction(self, self.config['action'])
        # self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()
    

    def _reward(self, action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError

    def _rewards(self, action) :
        """
        Returns a multi-objective vector of rewards.

        If implemented, this reward vector should be aggregated into a scalar in _reward().
        This vector value should only be returned inside the info dict.

        :param action: the last action performed
        :return: a dict of {'reward_name': reward_value}
        """
        raise NotImplementedError

    def _is_terminated(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        raise NotImplementedError

    def _is_truncated(self) -> bool:
        """
        Check we truncate the episode at the current step

        :return: is the episode truncated
        """
        raise NotImplementedError

    def _info(self, obs, action) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
        }
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        return info

    def reset(self,
              *,
              seed=0,
              options=None,
    ) :
        """
        Reset the environment to it's initial configuration

        :param seed: The seed that is used to initialize the environment's PRNG
        :param options: Allows the environment configuration to specified through `options["config"]`
        :return: the observation of the reset state
        """
        super().reset(seed=self.config['seed'], options=options)
        if options and "config" in options:
            self.configure(options["config"])
        self.update_metadata()
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.done = False
        self._reset()
        self.define_spaces()  # Second, to link the obs and actions to the vehicles once the scene is created
        # obs = self.observation_type.observe()
        # info = self._info(obs, action=self.action_space.sample())
        if self.render:
            self.display()
        # return obs, info
        

    def _reset(self) -> None:
        """
        Reset the scene: roads and vehicles.

        This method must be overloaded by the environments.
        """
        raise NotImplementedError()

    def step(self, action):
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        # obs = self.observation_type.observe()
        # reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        # info = self._info(obs, action)
        if self.render:
            self.display()

        # return obs, reward, terminated, truncated, info

    def _simulate(self, action) -> None:
        """Perform several steps of simulation with constant action."""
        # frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        # for frame in range(frames):
            # Forward action to the vehicle
        if action is not None \
                and not self.config["manual_control"] :
                # and self.steps % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
            self.action_type.act(action)

        self.road.act()
        # self.road.step(1 / self.config["simulation_frequency"])
        self.road.step(1/ self.config["policy_frequency"])
        self.steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            # if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                # self._automatic_rendering()

        self.enable_auto_render = False

    def display(self):
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        """
        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        # if self.render == 'rgb_array':
        #     image = self.viewer.get_image()
        #     return image

    def close(self) -> None:
        """
        Close the environment.

        Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def get_available_actions(self):
        return self.action_type.get_available_actions()

    def set_record_video_wrapper(self, wrapper):
        self._record_video_wrapper = wrapper
        self.update_metadata()

    def _automatic_rendering(self) -> None:
        """
        Automatically render the intermediate frames while an action is still ongoing.

        This allows to render the whole video and not only single steps corresponding to agent decision-making.
        If a RecordVideo wrapper has been set, use it to capture intermediate frames.
        """
        if self.viewer is not None and self.enable_auto_render:

            if self._record_video_wrapper and self._record_video_wrapper.video_recorder:
                self._record_video_wrapper.video_recorder.capture_frame()
            else:
                self.display()

    def simplify(self) -> 'AbstractEnv':
        """
        Return a simplified copy of the environment where distant vehicles have been removed from the road.

        This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, self.PERCEPTION_DISTANCE)

        return state_copy

    # def change_vehicles(self, vehicle_class_path: str) -> 'AbstractEnv':
    #     """
    #     Change the type of all vehicles on the road

    #     :param vehicle_class_path: The path of the class of behavior for other vehicles
    #                          Example: "highway_env.vehicle.behavior.IDMVehicle"
    #     :return: a new environment with modified behavior model for other vehicles
    #     """
    #     vehicle_class = utils.class_from_path(vehicle_class_path)

    #     env_copy = copy.deepcopy(self)
    #     vehicles = env_copy.road.vehicles
    #     for i, v in enumerate(vehicles):
    #         if v is not env_copy.vehicle:
    #             vehicles[i] = vehicle_class.create_from(v)
    #     return env_copy

    # def set_preferred_lane(self, preferred_lane: int = None) -> 'AbstractEnv':
    #     env_copy = copy.deepcopy(self)
    #     if preferred_lane:
    #         for v in env_copy.road.vehicles:
    #             if isinstance(v, IDMVehicle):
    #                 v.route = [(lane[0], lane[1], preferred_lane) for lane in v.route]
    #                 # Vehicle with lane preference are also less cautious
    #                 v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
    #     return env_copy

    # def set_route_at_intersection(self, _to: str) -> 'AbstractEnv':
    #     env_copy = copy.deepcopy(self)
    #     for v in env_copy.road.vehicles:
    #         if isinstance(v, IDMVehicle):
    #             v.set_route_at_intersection(_to)
    #     return env_copy

    # def set_vehicle_field(self, args: Tuple[str, object]) -> 'AbstractEnv':
    #     field, value = args
    #     env_copy = copy.deepcopy(self)
    #     for v in env_copy.road.vehicles:
    #         if v is not self.vehicle:
    #             setattr(v, field, value)
    #     return env_copy

    # def call_vehicle_method(self, args: Tuple[str, Tuple[object]]) -> 'AbstractEnv':
    #     method, method_args = args
    #     env_copy = copy.deepcopy(self)
    #     for i, v in enumerate(env_copy.road.vehicles):
    #         if hasattr(v, method):
    #             env_copy.road.vehicles[i] = getattr(v, method)(*method_args)
    #     return env_copy

    # def randomize_behavior(self) -> 'AbstractEnv':
    #     env_copy = copy.deepcopy(self)
    #     for v in env_copy.road.vehicles:
    #         if isinstance(v, IDMVehicle):
    #             v.randomize_behavior()
    #     return env_copy

    # def to_finite_mdp(self):
    #     return finite_mdp(self, time_quantization=1/self.config["policy_frequency"])

    # def __deepcopy__(self, memo):
    #     """Perform a deep copy but without copying the environment viewer."""
    #     cls = self.__class__
    #     result = cls.__new__(cls)
    #     memo[id(self)] = result
    #     for k, v in self.__dict__.items():
    #         if k not in ['viewer', '_record_video_wrapper']:
    #             setattr(result, k, copy.deepcopy(v, memo))
    #         else:
    #             setattr(result, k, None)
    #     return result



class IntersectionEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "acceleration_range": [-10, 10],
                "steering_range": [-np.pi / 3, np.pi / 3]
            },
            "duration": 13,  # [s]
            "destination": "o1",
            "controlled_vehicles": 1,
            "controlled_vehicles_batch":4,
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "screen_width": 900,
            "screen_height": 900,
            "centering_position": [0.5, 0.5],
            "scaling": 5,
            "collision_reward": -5,
            "high_speed_reward": 1,
            "arrived_reward": 1,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "offroad_terminal": False
        })
        return config


    def _is_terminated(self) -> bool:
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) \
               or (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (vehicle.crashed or
                self.has_arrived(vehicle))

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        return info

    def _reset(self) -> None:
        self._make_road()

    def step(self, action: int) :
        super().step(action)
        self._clear_vehicles()


    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        outer_distance = lane_width * 2.5
        access_length = lane_length  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            # priority = 3 if is_horizontal else 1
            priority = 0
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(f'o{corner}', f'ir{corner}',
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # Right turn
            right_turn_radius = lane_width * 2  # [m}
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane(f'ir{corner}', f'il{(corner - 1) % 4}',
                         right_turn_lane := CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            left_turn_radius = lane_width * 3  # [m}
            l_center = rotation @ (np.array([-outer_distance, outer_distance]))
            net.add_lane(f'ir{corner}', f'il{(corner + 1) % 4}',
                         left_turn_lane := CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # U turn
            y_center2 = 1.62467 * lane_width
            y_center2_from_outer = outer_distance - y_center2
            r2 = y_center2 / np.sqrt(2) - car_width
            r13 = 1.5 * lane_width
            switch_angle = np.arctan(y_center2_from_outer / outer_distance)
            center2 = rotation @ (np.array([0, y_center2]))
            center3 = rotation @ (np.array([-outer_distance, outer_distance]))
            net.add_lane(f'ir{corner}', f'im{corner}1',
                u_lane1 := CircularLane(r_center, r13, angle + np.radians(180), angle + np.radians(180) + switch_angle, line_types=[n, n], priority=priority, speed_limit=10),
            )
            net.add_lane(f'im{corner}1', f'im{corner}2',
                u_lane2 := CircularLane(center2, r2, angle + switch_angle, angle + np.radians(-180) - switch_angle, clockwise=False, line_types=[n, n], priority=priority, speed_limit=10),
            )
            net.add_lane(f'im{corner}2', f'il{corner}',
                u_lane3 := CircularLane(center3, r13, angle - switch_angle, angle, line_types=[n, n], priority=priority, speed_limit=10),
            )

            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(f'ir{corner}', f'il{(corner + 2) % 4}',
                         straight_lane := StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(f'il{(corner - 1) % 4}', f'o{(corner - 1) % 4}',
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

            corner = corner if corner % 2 == 0 else ((corner + 2) % 4) # Mapping HighwayEnv to external
            self.subzones[f'{corner}{corner}'].shape = [
                *right_turn_lane.position(np.linspace(0, right_turn_lane.length / 2, 10), lane_width / 2),
                *right_turn_lane.position(np.linspace(right_turn_lane.length / 2, 0, 10), -lane_width / 2),
            ]
            self.subzones[f'{corner}{(corner + 1) % 4}'].shape = [
                *right_turn_lane.position(np.linspace(right_turn_lane.length / 2, right_turn_lane.length, 10), lane_width / 2),
                *right_turn_lane.position(np.linspace(right_turn_lane.length, right_turn_lane.length / 2, 10), -lane_width / 2),
            ]
            self.subzones[f'{corner}{(corner + 2) % 4}'].shape = [
                *right_turn_lane.position(np.linspace(right_turn_lane.length / 2, right_turn_lane.length, 10), -lane_width / 2),
                straight_lane.position(straight_lane.length / 2, -lane_width / 2),
            ]
            self.subzones[f'{corner}{(corner + 3) % 4}'].shape = [
                *right_turn_lane.position(np.linspace(0, right_turn_lane.length / 2, 10), -lane_width / 2),
                straight_lane.position(straight_lane.length / 2, -lane_width / 2),
            ]

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def make_vehicle(self, initial_lane, target_lane, speed, **kwargs) -> Vehicle:
        """
        :return: the ego-vehicle
        """
        # Configure vehicles for IDM surrounding vehicles
        # vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        # vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        # vehicle_type.COMFORT_ACC_MAX = 6
        # vehicle_type.COMFORT_ACC_MIN = -3
        # assert controlled_veh_batch < 5, "controlled_veh_batch should be less than 5"

        # Controlled vehicles
        
        ego_lane = self.road.network.get_lane((f'o{initial_lane}', f'ir{initial_lane}', 0))
        destination = f'o{target_lane}'
    
        init_postion = 0
        ego_vehicle = self.action_type.vehicle_class(
                            self.road,
                            ego_lane.position(init_postion, 0),
                            speed=speed,
                            heading=ego_lane.heading_at(60))

        ### add attributes to enable external planning
        ego_vehicle.__dict__.update(**kwargs)

        
        try:
            ego_vehicle.plan_route_to(destination)
            ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
            ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
        except AttributeError:
            pass

        # Prevent early collisions
        self.road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        for v in self.road.vehicles:  
            if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                self.road.vehicles.remove(ego_vehicle)
                self.controlled_vehicles.remove(ego_vehicle)
                return None

        # bound added veh to action space
        self.define_spaces()
        
        return ego_vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH

        # self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
        #                       vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None or self.has_arrived(vehicle))]
        
        ### Remove controlled vehicles if it has arrived
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              not (is_leaving(vehicle) or vehicle.route is None or self.has_arrived(vehicle))]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 80) -> bool:
        return "il" in vehicle.lane_index[0] \
               and "o" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance


class MultiAgentIntersectionEnv(IntersectionEnv):
    def __init__(self, *args, subzones=None, **kwargs):
        self.subzones = subzones
        super().__init__(*args, **kwargs)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                 "type": "MultiAgentAction",
                 "action_config": {
                     "type": "AccOnlyAction",
                     "lateral": True,
                     "longitudinal": True
                 }
            },
            ## ignore the observation config
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }
            },         
        })
        return config


        