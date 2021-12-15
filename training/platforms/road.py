from enum import Enum, auto

import numpy as np
from training.platforms.car import CarLight
from training.platforms.helper import remove_first_edge, remove_first_node
from training.platforms.traffic_light import Direction, TrafficLight


class Road(object):
    def __init__(self, start: TrafficLight, end: TrafficLight, direction: Direction, speed_limit=40, lanes=1):

        self.start = start
        self.end = end
        self.vector = np.array([end.pos_x, end.pos_y]) - \
            np.array([start.pos_x, start.pos_y])
        self.set_direction(direction)
        self.speed_limit = speed_limit
        self.lanes = lanes

        self.pixels_per_mile = 270
        self.calc_length()
        self.calc_time()

        self.cars = []

    def set_direction(self, direction):
        if isinstance(direction, str):
            self.direction = Direction[direction]
        elif isinstance(direction, Direction):
            self.direction = direction

    def add_car(self, car: CarLight, position=0):
        "Take a car and add it to the list"

        assert len(car.path.nodes) == len(car.path.edges)

        car.current_road = self
        car.current_position = 0
        self.cars.append(car)

    def calc_length(self):
        x1, y1 = self.start.pos_x, self.start.pos_y
        x2, y2 = self.end.pos_x, self.end.pos_y

        self.length = np.linalg.norm(np.array([x2 - x1, y2 - y1]))

    def calc_time(self):
        self.time = self.length / self.speed_limit / self.pixels_per_mile

    def update(self, dt):
        """for all cars on the road update their positions.
        If they're at the end, put in the appropriate queue of the next light
        """
        indexes_to_move = []
        for i in range(len(self.cars)):
            current_pos = self.cars[i].current_position
            next_pos = current_pos + self.speed_limit * dt / \
                (60*60) * self.pixels_per_mile  # speed limit in mph, dt in seconds
            if next_pos > self.length:  # car has reached end of this road
                indexes_to_move.append(i)
            self.cars[i].current_position = next_pos

        # remove these indexes from list and add to light
        for i in indexes_to_move[::-1]:  # reverse list order
            car = self.cars.pop(i)

            car.path = remove_first_edge(car.path)
            self.end.add_car(car, dir_from=-self.direction)

    def __repr__(self):
        return "{}: {} mph".format(self.direction, self.speed_limit)

    def __add__(self, other):
        return self.time + other

    def __radd__(self, other):
        return other + self.time
