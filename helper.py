import numpy as np
import carla

def find_closest_waypoint(current_waypoint, route):
    """
    Find the closest waypoint to the current waypoint from the route (list of waypoints)
    :param current_waypoint: current waypoint
    :param route: route (list of waypoints)
    
    :return: closest waypoint, closest index
    """
    closest_waypoint = min(route, key=lambda x: x[0].transform.location.distance(current_waypoint.transform.location))
    closest_index = route.index(closest_waypoint)

    return closest_waypoint, closest_index

def coord2dist(coord):
    """
    Calculate the distance from the origin to the coordinate
    :param coord: coordinate

    :return: distance
    """
    return np.sqrt(coord.x**2 + coord.y**2) 

def camera_callback(image, data_dict):
    """
    Callback function for camera sensor

    Note: the height and width of the image need to be 2^n	
    """
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.width, image.height, 4))

def collision_callback(event, data_dict):
    """
    Callback function for collision sensor
    """
    data_dict['collision'] = True

def seg_camera_callback(image, data_dict):
    """
    Callback function for segmentation camera sensor
    
    Note: the height and width of the image need to be 2^n	
    """
    image.convert(carla.ColorConverter.CityScapesPalette)
    data_dict['sem_image'] = np.reshape(np.copy(image.raw_data), (image.width, image.height, 4))

def find_angle_error_vehicle2nextWaypoint(vehicle, next_waypoint):
    """
    Calculate the difference between the vehicle's yaw and the angle between the vehicle and the next waypoint
    This function is used for the follow the carrot controller

    :param vehicle: vehicle object
    :param next_waypoint: next waypoint (x, y, z, pitch, yaw, roll)

    :return: angle error
    """

    # Extract the current position and yaw of the vehicle
    current_pos_xy = [vehicle.get_location().x, vehicle.get_location().y]
    current_rot_yaw = vehicle.get_transform().rotation.yaw

    # Extract the next waypoint's position and yaw
    next_pos_xy = [next_waypoint.transform.location.x, next_waypoint.transform.location.y]
    next_rot_yaw = next_waypoint.transform.rotation.yaw

    # Calculate the angle between the vehicle and the next waypoint in radians
    angle = np.arctan2(next_pos_xy[1] - current_pos_xy[1], next_pos_xy[0] - current_pos_xy[0])

    # Calculate the difference in angles between the vehicle's yaw and the angle between the vehicle and the next waypoint
    angle_error = angle - current_rot_yaw*np.pi/180

    # Normalize the angle error between -pi and pi
    return (angle_error + np.pi) % (2 * np.pi) - np.pi