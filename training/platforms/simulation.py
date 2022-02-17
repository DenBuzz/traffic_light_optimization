
from os import stat
import random

import numpy as np
from dijkstar import find_path
from ray.rllib.utils.typing import MultiAgentDict
from training.generate_graph import generate_graph
from training.platforms.car import CarLight
from training.platforms.helper import remove_first_node


class Simulation(object):
    "Class to handle running the traffic simulation"

    def __init__(self, graph, config={}):
        self.set_default_config()

        if type(graph) is dict:  # received args for generate_graph
            self.graph_args = graph
        else:
            self.graph = graph

        self.random_cars = True
        self.random_car_probability = 0.05

        for attr, val in config.items():
            setattr(self, attr, val)

        self.cars = []

    def set_default_config(self):
        self.time = 0
        self.dt = 1
        self.graph_args = None
        # Trying to pull down rewards to something more reasonable
        self.reward_normalization = 1000

    def get_roads(self):
        "get all roads in the sim"
        self.roads = []
        for light in self.graph:
            for road in self.graph[light].values():
                self.roads.append(road)

    def loop(self, total_time=1000):
        "Loop to run the simulation"

        # ToDo generate cars in a dope way
        self.generate_cars()

        running = True
        while running:
            self.update_sim()

    def update_sim(self, steps=1, action: MultiAgentDict = {}):
        for i in range(steps):
            self.time += self.dt

            if self.random_cars:
                if random.random() < self.random_car_probability:
                    self.generate_cars(1)

            for road in self.roads:
                road.update(dt=self.dt)

            for light in self.lights:
                light.update(dt=self.dt, action=action[light.light_id])

            for car in self.cars:
                car.update(dt=self.dt)
                if car.arrived:
                    self.remove_car(car)

    def reset(self):
        "Reset the sim data"
        if self.graph_args is not None:
            self.graph = generate_graph(**self.graph_args)

        self.lights = list(self.graph.get_data().keys())
        for i, light in enumerate(self.lights):
            light.light_id = 'traffic_light_{}'.format(i)

        self.get_roads()

        self.time = 0
        self.cars = []
        for light in self.lights:
            light.empty_queues()

    def generate_cars(self, num=1):
        for i in range(num):
            start_light, end_light = random.sample(self.lights, 2)
            path = find_path(self.graph, start_light, end_light)

            path = remove_first_node(path)  # Remove first node
            # print('Path: {}'.format(path))

            car = CarLight(start_light, end_light, path)
            first_road = path.edges[0]

            self.cars.append(car)
            first_road.add_car(car)

    def remove_car(self, car):
        self.cars.remove(car)

    def get_features(self):
        data = {}
        for light in self.lights:
            features = np.array(light.get_features())
            queue_features = features[:12]**(1/3) - 1
            state_feature = features[12:]
            data[light.light_id] = np.append(queue_features, state_feature)
        return data

    def get_rewards(self):
        rewards = {}
        for light in self.lights:
            reward = light.get_reward() / self.reward_normalization
            rewards[light.light_id] = -reward  # make it a penalty!
        return rewards
