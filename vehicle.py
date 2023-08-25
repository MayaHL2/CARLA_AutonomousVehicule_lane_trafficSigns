import carla
import time
import random
from helper import *

class Vehicle():
    """
    Class to create the vehicle and its sensors
    """
    def __init__(self, world):
        self.world = world
        self.sensors = dict()
        self.sensor_data = dict()

        self.camera = None
        self.seg_camera = None
        self.collision_sensor = None

    def spawn_vehicle(self, vehicle_name = None, spawn_point = None):
        """
        Function to spawn the vehicle

        :param vehicle_name: name of the vehicle to spawn
        :param spawn_point: where to spawn the vehicle

        :return: vehicle: vehicle spawned
        """
        if vehicle_name == None or vehicle_name not in [bp.id for bp in self.world.get_blueprint_library().filter('vehicle')]:
            vehicle_name = 'vehicle.lincoln.mkz_2020'
        if spawn_point == None:
            spawn_point = random.choice(self.world.get_map().get_spawn_points())

        vehicle_bp = self.world.get_blueprint_library().find(vehicle_name)
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

        # Place the spectator on the vehicle (x=0.8, z=1.4) from the vehicle's perspective
        spectator = self.world.get_spectator()
        transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=0.8, z=1.4)), self.vehicle.get_transform().rotation)
        spectator.set_transform(transform)

        return self.vehicle

    def spawn_camera(self, disp_size = [256, 256], relative_location = carla.Location(x=0.8, z=1.4), fov = 90):
        """
        Function to add the camera to the vehicle

        :param disp_size: size of the image
        :param relative_location: location of the camera relative to the vehicle
        :param fov: field of view of the camera

        :return: camera: camera spawned
        """
        
        self.sensor_data['image'] = np.zeros((disp_size[0], disp_size[1], 4))

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(disp_size[0]))
        camera_bp.set_attribute('image_size_y', str(disp_size[1]))
        camera_bp.set_attribute('fov', str(fov))
        camera_init_translation = carla.Transform(relative_location, carla.Rotation())
        camera = self.world.spawn_actor(camera_bp, camera_init_translation, attach_to=self.vehicle)
        time.sleep(0.2)

        self.camera = camera
        self.sensors["camera"] = [camera, camera_callback]

        return camera

    def spawn_semantic_camera(self, disp_size = [256, 256], relative_location = carla.Location(x=0.8, z=1.4)):
        """
        Function to add the segmentation camera to the vehicle

        :param disp_size: size of the image
        :param relative_location: location of the segmentation camera relative to the vehicle
        :param fov: field of view of the camera

        :return: camera: camera spawned
        """
        self.sensor_data['sem_image'] = np.zeros((disp_size[0], disp_size[1], 4))

        seg_camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', str(disp_size[0]))
        seg_camera_bp.set_attribute('image_size_y', str(disp_size[1]))
        seg_camera_init_translation = carla.Transform(relative_location, carla.Rotation())
        seg_camera = self.world.spawn_actor(seg_camera_bp, seg_camera_init_translation, attach_to=self.vehicle)
        time.sleep(0.2)

        self.seg_camera = seg_camera
        self.sensors["seg_camera"] = [seg_camera, seg_camera_callback]

        return seg_camera

    def spawn_collision_sensor(self):
        """
        Function to add the collision sensor to the vehicle

        :return: collision_sensor: collision sensor spawned
        """
        self.sensor_data['collision'] = False

        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        time.sleep(0.2)

        self.collision_sensor = collision_sensor
        self.sensors["collision_sensor"] = [collision_sensor, collision_callback]

        return collision_sensor

    def start_listen(self): 
        """
        Function to start listening to the sensors
        """
        if self.camera != None:
            self.camera.listen(lambda image: camera_callback(image, self.sensor_data))
        if self.seg_camera != None:
            self.seg_camera.listen(lambda image: seg_camera_callback(image, self.sensor_data))
        if self.collision_sensor != None:
            self.collision_sensor.listen(lambda event: collision_callback(event, self.sensor_data))
    
    def stop_sensors(self):
        """
        Stop all the sensors
        """
        for sensor in self.sensors.values():
            sensor[0].stop()
            sensor[0].destroy()

    def destroy_vehicle(self):
        self.vehicle.destroy()

    def get_spectator(self):
        return self.world.get_spectator()
    
    def get_sensor_data(self):
        return self.sensor_data
    
    def get_velocity(self):
        return self.vehicle.get_velocity()
    
    def get_current_waypoint(self):
        return self.world.get_map().get_waypoint(self.vehicle.get_location())
    
    def get_transform(self):
        return self.vehicle.get_transform()
    
    def get_location(self):
        return self.vehicle.get_location()
    
    def apply_control(self, control):
        self.vehicle.apply_control(control)