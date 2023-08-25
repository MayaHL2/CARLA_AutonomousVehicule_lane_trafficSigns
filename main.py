import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
from helper import *
from navigation import Navigation
from vehicle import Vehicle
from lane_detection_annotation import *

from navigation import IM_WIDTH, IM_HEIGHT

import cv2
import matplotlib
matplotlib.use('TkAgg')

random.seed(1)

try : 
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
except ValueError:
    raise("client couldn't connect")

actor_list = []

scalar = 1
disp_size = [IM_WIDTH,IM_HEIGHT]*scalar

collision_flag = False

try : 
    # world = client.get_world()
    world = client.load_world('Town10HD')
    map = world.get_map()

    # Change the weather to ClearNight, ClearNoon ...
    world.set_weather(carla.WeatherParameters.ClearNoon)

    car = Vehicle(world)

    vehicle = car.spawn_vehicle()
    camera = car.spawn_camera(disp_size=disp_size)
    sem_camera = car.spawn_semantic_camera(disp_size=disp_size)
    collision_sensor = car.spawn_collision_sensor()

    car.start_listen()

    spectator = car.get_spectator()

    sensor_data = car.get_sensor_data()

    # Choose random point on the map
    start = vehicle.get_location()
    goal = random.choice(map.get_spawn_points()).location

    # Set the desired speed in km/h
    desired_speed = 30
    
    Nav = Navigation(map, vehicle, spectator, start, goal, desired_speed, sensor_data = sensor_data, sensors = {'camera': camera, "collision_sensor": collision_sensor, "sem_camera": sem_camera})
    Nav.run_navigation(5, display_cameras = True, save_data= True, camera_step_save= 0.05, save_video= True, time_limit= 60)

    # Nav.draw_route()
    # Nav.draw_graph_error("speed_error")

    car.stop_sensors()
    car.destroy_vehicle()

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')