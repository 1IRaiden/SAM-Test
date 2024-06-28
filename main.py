import time
import numpy as np
from Drone import Drone, CurrentsDrone
from connection_for_camera import Camera, Position
from Multithread import Multithreading
import threading
import numpy



# variable for main
count_drones = 2

start_position = np.array([[1, 2, 3],
              [-1, 0, 0]
              ], dtype=np.float32)

start_health = [100, 75]

FIRST_TRANSLATION = True

def main(queue):
    global FIRST_TRANSLATION
    while True:
        if not th_setting.is_alive():
            if FIRST_TRANSLATION:
                th_videostream.start()
                time.sleep(6)
                FIRST_TRANSLATION = False
            with lock:
                queue.put(current_drone.all_drones)

            position = pos.increase_value()
            for i, drone in enumerate(current_drone):
                drone.update_position(position[i])
        time.sleep(0.1)


if __name__ == '__main__':
    camera = Camera()
    pos = Position()
    queue = Multithreading()
    lock = threading.Lock()

    current_drone = CurrentsDrone()

    for i in range(0, count_drones):
        drone = Drone(start_position[i], start_health[i])
        current_drone.all_drones.append(drone)

    print(type(current_drone))
    queue.put(current_drone.all_drones[0])

    th_videostream = threading.Thread(target=camera.start_translation, args=(queue.qu,))
    th_setting = threading.Thread(target=camera.setting_camera)
    th_get_info_drones = threading.Thread(target=main, args=(queue.qu,))

    th_setting.start()
    th_get_info_drones.start()















