from dijkstar.algorithm import PathInfo
from enum import Enum, auto
import numpy as np


class Direction(Enum):
    N = 1
    E = 2
    S = -1
    W = -2

    @property
    def vector(self):
        if self.name == 'N':
            return np.array([0, -1])
        if self.name == 'S':
            return np.array([0, 1])
        if self.name == 'E':
            return np.array([1, 0])
        if self.name == 'W':
            return np.array([-1, 0])

    @property
    def angle(self):
        if self.name == 'N':
            return 0
        if self.name == 'S':
            return 180
        if self.name == 'E':
            return 90
        if self.name == 'W':
            return 270

    def __neg__(self):
        return Direction(-self.value)


class LightState(Enum):
    NS = auto()
    EW = auto()
    NS_L = auto()
    EW_L = auto()


def remove_first_node(path: PathInfo):
    "removes the first node from the path"
    return path._replace(nodes=path.nodes[1:])


def remove_first_edge(path: PathInfo):
    "removes the first edge from the path"
    return path._replace(edges=path.edges[1:])
