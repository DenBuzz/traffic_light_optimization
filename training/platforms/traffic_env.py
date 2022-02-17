import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from training.platforms.simulation import Simulation
from training.platforms.render import Window
from training.platforms.helper import LightState
from training.generate_graph import generate_graph
from ray.rllib.utils.typing import MultiAgentDict
from ray.rllib.env import MultiAgentEnv
from gym.spaces import Box, Discrete
import numpy as np
from typing import Tuple
import random


class TrafficEnv(MultiAgentEnv):
    """Environment to run the traffic simulation"""

    def __init__(self, config):

        self.render_initialized = False

        self.set_default_config()

        for attr, val in config.items():
            setattr(self, attr, val)

        assert self.graph is not None
        self.sim = Simulation(self.graph)

        self.sim.random_cars = self.sim_random_cars
        self.sim.random_car_probability = self.sim_random_car_probability
        self.sim.dt = self.sim_dt
        self.observation_space = Box(
            low=-2, high=1000, shape=(12,), dtype=float)
        self.action_space = Discrete(4)

    def set_default_config(self):
        self.graph = None
        self.sim_random_cars = True
        self.sim_random_car_probability = 0.05
        self.sim_dt = 1
        self.steps_per_action = 10
        self.episode_length = 86400

    def reset(self) -> MultiAgentDict:

        self.sim.reset()

        initial_states = {}
        for light in self.sim.lights:
            initial_states[light.light_id] = 1
        self.sim.update_sim(steps=self.steps_per_action, action=initial_states)

        features = self.sim.get_features()

        return features

    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        self.sim.update_sim(steps=self.steps_per_action, action=action_dict)
        features = {}
        rewards = {}
        dones = {'__all__': False}
        infos = {}

        features = self.sim.get_features()
        rewards = self.sim.get_rewards()

        # Check if the episode is complete
        if self.sim.time >= self.episode_length:
            dones['__all__'] = True

        for agent in features:
            infos[agent] = {}
            dones[agent] = False

        return features, rewards, dones, infos

    def render(self, mode=None) -> None:
        if not self.render_initialized:
            self.initialize_render()

        self.window.draw_sim()

        return super().render(mode=mode)

    def initialize_render(self):
        self.render_initialized = True
        self.window = Window(self.sim, config={})


if __name__ == '__main__':

    print('Testing env...')

    from time import sleep, time

    from training.generate_graph import generate_graph

    # graph = generate_graph((5, 5), lights=12)

    env = TrafficEnv({'graph': {'grid_size': (5, 4), 'lights': 14}, "steps_per_action": 1, 'sim_dt': 0.1,
                     'sim_random_car_probability': 0.02, 'episode_length': 1000})

    episode_rewards = []
    for i in range(3):
        data = env.reset()
        dones = {'__all__': False}

        action_counter = 0
        action_per = 500
        actions = {}
        for light in data:
            actions[light] = random.randint(0, 3)

        episode_reward = 0
        start_time = time()
        while not dones['__all__']:
            # for light in data:
            #     print(data[light])
            action_counter += 1
            if action_counter >= action_per:
                for light in data:
                    actions[light] = random.randint(0, 3)
                action_counter = 0

            data, reward, dones, info = env.step(actions)
            env.render()
            episode_reward += sum(reward.values())
            # print(reward.values())
            # sleep(0.01666)
            sleep(0.001) # 100x Real time
        end_time = time()
        episode_rewards.append(episode_reward)

    print('Test complete')
    print('Total time: {}'.format(end_time - start_time))
    print('Mean Episode Reward: {}'.format(np.mean(episode_rewards)))
