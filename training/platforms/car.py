from dijkstar.algorithm import PathInfo


class CarRoad():
    def __init__(self, start_road, end_road, start_pos, end_pos, path=None):
        self.start_road = start_road
        self.end_road = end_road
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.path = path

        self.current_road = start_road

        self.arrived = False

    def turn_on_road(self, road):
        self.current_road = road
        self.current_pos = 0

    def update_pos(self, time=1/24):
        "update the position of the car along the road"
        speed = self.current_road.speed_limit

        new_pos = self.current_pos + speed * time

        if self.current_road == self.end_road and self.current_pos >= self.end_pos:
            # we have made it to our destination!
            self.arrived = True

        # todo: if past end of road, add to queue for the next light.


class CarLight():
    def __init__(self, start, end, path: PathInfo):
        self.start = start
        self.end = end
        self.path = path
        self.trip_duration = 0
        self.at_light = False
        self.light_time = 0

        self.arrived = False

        self.current_road = None
        self.current_position = 0

    def update(self, dt):
        self.trip_duration += dt
        if self.at_light:
            self.light_time += dt

    def arrived_at_light(self):
        self.at_light = True
        self.light_time = 0

    def leaving_light(self):
        self.at_light = False
        self.light_time = 0
