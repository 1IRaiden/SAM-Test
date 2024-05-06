import json
import numpy
from Order import Circle, Rectangle, Any


# circle
# rectangle

class Drone:

    name_json_file = "data.json"
    data = {}

    __DATA_READ = False

    def __init__(self, position, health=100):
        self.position = position
        self.health = health

        if not Drone.__DATA_READ:
            Drone.read_data()

        self.circle = Circle(Drone.data['circle'])
        self.rectangle = Rectangle(Drone.data['rectangle'])

        self.any = Any('Not important/new_image.jpg', 'Not important/mask_inv.png', Drone.data['circle'])

    @staticmethod
    def read_data():
        with open(Drone.name_json_file, "r") as file:
            Drone.data = json.load(file)
        Drone.DATA_READ = True

        Drone.data["circle"]["color"] = tuple(Drone.data["circle"]["color"])
        Drone.data["rectangle"]["color"] = tuple(Drone.data["rectangle"]["color"])

    def update_position(self, position):
        self.position = position

    def update_health(self, value):
        self.health = value
        self.rectangle.count_dx_h(value, self.health)


class CurrentsDrone:
    def __init__(self, count=0):
        self.all_drones = []

    def __iter__(self):
        return iter(self.all_drones)


if __name__ == "__main__":
    drone = Drone((0, 0, 0), 50)
    print(drone.circle.setting)
    print(drone.rectangle.setting)


