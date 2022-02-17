import os
import sys

from dijkstar import Graph

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import numpy as np

from training.platforms.road import Road
from training.platforms.traffic_light import TrafficLight


def generate_graph(grid_size: tuple = (10, 10), lights: int = 4):
    """function to generate graphs to train with
    Kinda sketchy approach but I think this should work.
    Needs some good comments
    """
    grid = np.zeros(grid_size)

    while sum(grid.flatten()) < lights:
        new_light = tuple(np.random.randint(grid_size))
        # print(f'New light at {new_light}')
        grid[new_light] = 1
        lights_to_check = [new_light, ]

        "add lights until we have at least the requested number"
        for light in lights_to_check:
            while count_roads_off_light(light, grid) < 2:
                light_added = branch_light(light, grid)
                grid[light_added] = 1
                lights_to_check.append(light_added)

    print(grid)
    print(f'Added {sum(grid.flatten())} lights...')
    graph = grid_to_graph(grid)
    return graph


def count_roads_off_light(light: tuple, grid: np.array):
    "count how many roads are already connected to the light"
    # print(grid)
    row = grid[light[0], :]
    col = grid[:, light[1]]
    row_right = 1 if sum(row[light[1] + 1:]) >= 1 else 0
    row_left = 1 if sum(row[:light[1]]) >= 1 else 0
    col_down = 1 if sum(col[light[0] + 1:]) >= 1 else 0
    col_up = 1 if sum(col[:light[0]]) >= 1 else 0
    total = row_left + row_right + col_up + col_down
    # print(f'Counting roads for {light}: {total}')
    return total


def branch_light(light: tuple, grid: np.array) -> tuple:
    "add a light branching off the one given in a random direction"
    # 0,1 choose which index of the light tuple
    coordinate_to_replace = np.random.randint(2)
    possible = set(range(grid.shape[coordinate_to_replace]))
    possible.remove(light[coordinate_to_replace])
    new_light = np.array(light)
    new_light[coordinate_to_replace] = np.random.choice(list(possible))
    # print(f'Branching new light at {new_light}')
    return tuple(new_light)


def grid_to_graph(grid: np.array):
    "Take the grid and convert it to a graph"
    light_coords = get_lights(grid)
    light_map = {(row, col): TrafficLight((row+1)*100, (col+1)*100) for row, col in light_coords}
    road_pairs = get_road_pairs(grid)
    print(f'Found {len(road_pairs)} roads')

    graph = Graph(undirected=False) 

    for road_pair in road_pairs:
        # Create the roads connecting all the lights
        first, second = road_pair
        road_len = ((first[0]-second[0])**2 + (first[1]-second[1])**2)**0.5
        speed = min([10 + 15 * road_len, 60])
        if first[1] != second[1]: # E, W
            if first[1] > second[1]: # W
                road1 = Road(light_map[first], light_map[second], 'W', speed)
                road2 = Road(light_map[second], light_map[first], 'E', speed)
            else: # E
                road1 = Road(light_map[second], light_map[first], 'W', speed)
                road2 = Road(light_map[first], light_map[second], 'E', speed)
        elif first[0] > second[0]: # N
            road1 = Road(light_map[first], light_map[second], 'N', speed)
            road2 = Road(light_map[second], light_map[first], 'S', speed)
        else: # S
            road1 = Road(light_map[second], light_map[first], 'N', speed)
            road2 = Road(light_map[first], light_map[second], 'S', speed)

        graph.add_edge(road1.start, road1.end, road1)
        graph.add_edge(road2.start, road2.end, road2)

    return graph


def get_lights(grid: np.array):
    "Get the list of light coords from the grid"
    light_coords = np.where(grid==1)
    light_coords = [(row, col) for row, col in zip(light_coords[0], light_coords[1])]
    return light_coords

def get_road_pairs(grid: np.array):
    "return pairs of coords where roads can be connected"
    road_pairs = []
    for i, row in enumerate(grid):
        lights = np.where(row == 1)[0]
        for j in range(len(lights) - 1):
            road_pairs.append(((i, lights[j]),(i, lights[j+1])))

    for j in range(grid.shape[1]):
        # iterate over the cols
        col = grid[:, j]
        lights = np.where(col == 1)[0]
        for i in range(len(lights)- 1):
            road_pairs.append(((lights[i], j), (lights[i+1], j)))
    
    return road_pairs


def main():
    generate_graph(grid_size=(5, 5), lights=10)


if __name__ == '__main__':
    main()
