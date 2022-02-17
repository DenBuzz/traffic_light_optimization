import random
import numpy as np
from collections import deque
from itertools import cycle

from training.platforms.car import CarLight
from training.platforms.helper import Direction, LightState, remove_first_node


class TrafficLight(object):
    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.light_id = 'traffic_light'

        # queue of cars waiting from the north
        self.queues = {}
        self.queue_delays = {}
        self.initialize_queues()

        self.in_roads = {}
        self.out_roads = {}

        self.cycle_timer = 0
        self.cycle_duration = 60  # 1 minute
        self.cycle_delay_timer = 0
        self.cycle_delay = 7  # 5 second yellow plus 2
        self.states = list(LightState)
        self.current_state = random.choice(self.states)
        self.delaying = False

        self.time_per_car = 2  # seconds

    def add_car(self, car: CarLight, dir_from: Direction):
        "Add new car from incoming road"
        assert len(car.path.nodes) == len(car.path.edges) + 1

        if len(car.path.edges) == 0:
            car.arrived = True
            # print('Trip duration: {}'.format(car.trip_duration))

        else:
            next_dir = car.path.edges[0].direction
            queue_label = dir_from.name + next_dir.name

            self.queues[queue_label].append(car)
            car.arrived_at_light()

    def update(self, dt, action):
        "Light can be in 4 possible states, NS, EW, NS_L, EW_L"
        # NS draws from NS, SN, SE, NW

        action_state = LightState(action+1)  # states are 1,2,3,4
        self.change_state(action_state)

        if self.delaying:
            self.cycle_delay_timer += dt
            if self.cycle_delay_timer >= self.cycle_delay:
                self.delaying = False
                self.cycle_delay_timer = 0
                self.current_state = self.next_state

        if not self.delaying:
            live_queues = []
            if self.current_state == LightState.NS:
                live_queues = ['NS', 'SN', 'SE', 'NW']
            elif self.current_state == LightState.EW:
                live_queues = ['EW', 'WE', 'EN', 'WS']
            elif self.current_state == LightState.NS_L:
                live_queues = ['SW', 'NE']
            elif self.current_state == LightState.EW_L:
                live_queues = ['WN', 'ES']

            for label in live_queues:
                q = self.queues[label]
                current_delay_timer = self.queue_delays[label]
                if current_delay_timer >= self.time_per_car and len(q) != 0:
                    self.queue_delays[label] = 0
                    car = q.popleft()
                    if len(car.path.edges) != 0:  # Not at destination yet
                        road = car.path.edges[0]
                        car.path = remove_first_node(car.path)
                        car.leaving_light()
                        road.add_car(car, position=0)

                elif len(q) != 0:
                    self.queue_delays[label] += dt

    def empty_queues(self):
        self.initialize_queues()

    def get_road(self, direction):
        "return the outgoing road of the light"
        pass

    def add_outgoing_road(self, road):
        "add outgoing roads to the light"
        self.out_roads[road.direction] = road

    def change_state(self, state: LightState):
        if state != self.current_state:
            self.delaying = True
            self.next_state = state

    def add_incoming_road(self, road):
        self.in_roads[road.direction] = road

    def green_NS(self):
        self.current_state = LightState.NS

    def green_EW(self):
        self.current_state = LightState.EW

    def green_NS_L(self):
        self.current_state = LightState.NS_L

    def green_EW_L(self):
        self.current_state = LightState.EW_L

    def initialize_queues(self):
        dirs = ['N', 'E', 'S', 'W']

        for d in dirs:
            for f in dirs:
                if f != d:
                    self.queues[d+f] = deque(maxlen=1000)
                    self.queue_delays[d+f] = 0

    def get_features(self):
        queue_lengths = []
        for queue in self.queues.values():
            queue_lengths.append(len(queue))
        current_state = self.current_state.value
        state_feature = np.zeros(4)
        state_feature[current_state-1] = 1
        queue_lengths.extend(state_feature)
        return queue_lengths

    def get_reward(self):
        current_penalty = 0
        for queue in self.queues.values():
            for car in queue:
                current_penalty += car.light_time
        return current_penalty

    def __repr__(self):
        return "TrafficLight at {}, {}".format(self.pos_x, self.pos_y)
