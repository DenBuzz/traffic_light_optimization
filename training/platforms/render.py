import random
from time import time

import numpy as np
import pygame
from training.platforms.car import CarLight
from training.platforms.helper import Direction


class Window:
    def __init__(self, sim, config={}):
        self.sim = sim
        self.set_default_config()

        for attr, val in config.items():
            setattr(self, attr, val)

        pygame.init()
        self.font = pygame.font.SysFont('monospace', 20, bold=True)
        window = (self.width, self.height)
        self.screen = pygame.display.set_mode(window)
        self.background = pygame.Surface(window)

    def set_default_config(self):
        self.width = 600
        self.height = 400
        self.fps = 60
        self.bg_color = (230, 230, 230)

        self.road_offset = 2
        self.road_width = 5
        self.road_color = (150, 150, 150)

        self.car_color = (0, 0, 150)

        self.light_color = (100, 100, 100)
        self.traffic_light_width = 20
        self.light_state_thickness = 3

    def loop(self):

        pygame.display.flip()

        running = True
        clock = pygame.time.Clock()
        while running:
            self.draw_background()
            self.screen.blit(self.background, (0, 0))
            for i in pygame.event.get():
                if i.type == pygame.QUIT:
                    running = False

            self.sim.update_sim()
            self.draw()

            pygame.display.flip()
            clock.tick(self.fps)

    def draw_sim(self):
        "to use when training with render. Just draw individual frames"
        self.draw_background()
        self.screen.blit(self.background, (0, 0))
        self.draw()
        self.draw_car_count()
        pygame.display.flip()

    def draw(self):
        self.draw_cars()
        self.draw_lights()

    def draw_background(self):
        self.background.fill(self.bg_color)
        self.draw_roads()
        self.draw_lights()

    def draw_roads(self):
        graph = self.sim.graph
        for light in graph:
            for road in graph[light].values():
                self.draw_road(road)

    def draw_lights(self):
        graph = self.sim.graph
        for light in graph:
            self.draw_light(light)

    def draw_light(self, light):
        x = light.pos_x - self.traffic_light_width / 2
        y = light.pos_y - self.traffic_light_width / 2

        pygame.draw.rect(self.background, self.light_color, (x, y,
                         self.traffic_light_width, self.traffic_light_width))

        light_state = light.current_state
        center = np.array([light.pos_x, light.pos_y])
        tlw = self.traffic_light_width

        for direction in Direction:
            if 'L' not in light_state.name:  # not turning left
                base_polygon = np.array([
                    [-tlw/2, -tlw/2],
                    [0, -tlw/2],
                    [0, -tlw/2 + self.light_state_thickness],
                    [-tlw/2, -tlw/2 +
                        self.light_state_thickness]
                ])
                if direction.name in light_state.name:
                    color = (0, 200, 0)
                else:
                    color = (200, 0, 0)

                rotated_polygon = self._rotate_polygon(
                    base_polygon, angle=direction.angle)

                poly = center + rotated_polygon
                pygame.draw.polygon(self.screen, color, poly)

            else:  # we are turning left!
                if direction.name in light_state.name:
                    color = (0, 200, 0)

                    if direction.name == 'N':
                        displacement = tlw * np.array([0, -1])
                    if direction.name == 'S':
                        displacement = tlw * np.array([-1, 0])
                    if direction.name == 'E':
                        displacement = tlw * np.array([0, 0])
                    if direction.name == 'W':
                        displacement = tlw * np.array([-1, -1])

                    x, y = center + displacement

                    start_angle = (-direction.angle + 180) * np.pi/180
                    end_angle = (-direction.angle + 270) * np.pi/180

                    pygame.draw.arc(self.screen, color,
                                    (x, y, tlw, tlw), start_angle, end_angle, width=2)

    def draw_road(self, road):
        start_light = road.start
        end_light = road.end

        start = np.array([start_light.pos_x, start_light.pos_y])
        end = np.array([end_light.pos_x, end_light.pos_y])

        vector = end - start
        u_vec_rotated = self._rotate_vec(vector)

        coord1 = start + self.road_offset * u_vec_rotated
        coord2 = end + self.road_offset * u_vec_rotated
        coord3 = coord2 + self.road_width * u_vec_rotated
        coord4 = coord1 + self.road_width * u_vec_rotated

        pygame.draw.polygon(self.background, self.road_color, [
                            coord1, coord2, coord3, coord4])

    def draw_cars(self):
        for car in self.sim.cars:
            self.draw_car(car)

    def draw_car(self, car: CarLight):

        if not car.arrived:

            road = car.current_road

            start = np.array([road.start.pos_x, road.start.pos_y])
            end = np.array([road.end.pos_x, road.end.pos_y])

            u_vec_rotated = self._rotate_vec(road.vector)

            coord = start + (end-start) * car.current_position / \
                road.length + (self.road_offset +
                               self.road_width/2) * u_vec_rotated

            pygame.draw.circle(self.screen, self.car_color,
                               (coord[0], coord[1]), 3)

    def draw_car_count(self):
        sim_time = self.sim.time
        time_label = self.font.render(
            'Sim time:    {}'.format(sim_time), True, (0, 0, 0))
        count = len(self.sim.cars)
        car_count_label = self.font.render(
            'Cars in sim: {}'.format(count), True, (0, 0, 0))
        self.screen.blit(time_label, (0, 0))
        self.screen.blit(car_count_label, (0, 20))

    def _rotate_vec(self, vector):
        "nomralized rotated vector 90 degrees clockwise (x, y) -> (-y, x) because +y is down"
        u_vec = vector / np.linalg.norm(vector)
        rot_vec = np.array([-u_vec[1], u_vec[0]])
        return rot_vec

    def _rotate_polygon(self, points, angle=0):
        angle = angle * np.pi/180
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)],
             [np.sin(angle), np.cos(angle)]]
        )
        new_points = []
        for point in points:
            new_point = np.matmul(rotation_matrix, point)
            new_points.append(new_point)

        return new_points
