from agents.navigation.global_route_planner import GlobalRoutePlanner
from helper import *
import carla
import cv2
import time
import matplotlib.pyplot as plt 
from TS_detection import *
from lane_detection_annotation import *

import threading as th

IM_HEIGHT = 512
IM_WIDTH = 512

class Navigation:
    """
    Class to control the vehicle and to detect the traffic signs and the lanes
    """
    def __init__(self, map, vehicle, spectator, start, goal, desired_speed, sensor_data, sensors = dict(), sampling_resolution = 2, max_throttle = 0.75):
        # Start and goal positions
        self.start = start
        self.goal = goal
        self.path = []

        # Define the max throttle
        self.max_throttle = max_throttle

        # Define the sensors and vehicle
        self.vehicle = vehicle
        self.sensors = sensors
        self.spectator = spectator

        # Define the best path to follow
        self.sampling_resolution = sampling_resolution
        route_planner = GlobalRoutePlanner(map, sampling_resolution)
        self.route = route_planner.trace_route(self.start, self.goal)

        # # Define the waypoints of the path
        path_waypoints = map.generate_waypoints(2)
        self.path_waypoints = path_waypoints

        # Generate the lanes
        self.left_lanes, self.right_lanes = generate_lanes(path_waypoints)
        lane_points = waypointsLane2array(self.left_lanes, self.right_lanes)
        lane_points = classify_lanes(lane_points, max_distance=10, max_angle=8)
        # Create the array contaning the lane points and their class
        self.lane_points = lane2points_class(lane_points)
        # Create the projection matrix
        self.K = build_projection_matrix(IM_WIDTH, IM_HEIGHT)

        # Define the colors for ploting the lanes
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0), (255, 255, 255), (128, 0, 0), (139, 0, 0), (178, 34, 34), (255, 69, 0), (255, 215, 0), (255, 255, 0), (128, 128, 0), (0, 255, 0), (124, 252, 0), (173, 255, 47), (127, 255, 0), (127, 255, 212), (72, 209, 204), (0, 255, 255), (0, 191, 255), (138, 43, 226), (255, 0, 255), (218, 112, 214), (221, 160, 221), (216, 191, 216), (128, 0, 128), (147, 112, 219), (139, 0, 139), (75, 0, 130), (255, 105, 180), (255, 127, 80), (255, 140, 0), (244, 164, 96), (210, 180, 140)]

        # Convert desired speed from km/h to m/s
        self.desired_speed = desired_speed/3.6

        # Define PID controllers
        self.PID_long = {"Kp": 1,
                        "Ki": 0.001,
                        "Kd": 10
                        }
        self.PID_lat = {"Kp": 0.7,
                        "Ki": 0,
                        "Kd": 10
                        }
        
        # Initialize the control loop variables
        self.image_number = 0
        self.last_speed_error = 0
        self.int_speed_error = 0
        self.last_angle_error = 0
        self.int_angle_error = 0

        # Initialize the graphs
        self.graphs_error = {"distance_error": [], "speed_error": [], "angle_error": []}
        self.graphs_control = {"throttle": [], "steer": [], "brake": []}
        self.graphs_values = {"speed": [], "angle": []}
        self.graph_route = []

        self.time = time.time()

        # Traffic signs detection initialization
        self.detection_graph, self.category_index = initialize_model()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)
        time.sleep(5)

        # Initialize the sensor data
        self.sensor_data = sensor_data

        self.run_step = True

        self.data_tiled = [None]

    def PID_coeff_long(self, Kp, Ki, Kd):
        """
        Function to define the PID coefficients for longitudinal control
        """

        self.PID_long["Kp"] = Kp
        self.PID_long["Ki"] = Ki
        self.PID_long["Kd"] = Kd

    # Define the PID controllers for lateral control
    def PID_coeff_lat(self, Kp, Ki, Kd):
        """
        Function to define the PID coefficients for lateral control
        """
        self.PID_lat["Kp"] = Kp
        self.PID_lat["Ki"] = Ki
        self.PID_lat["Kd"] = Kd

    def longitudinal_control(self):
        """
        Function to calculate the longitudinal control, i.e. the throttle and the brake

        :return: throttle, brake
        """
        # Calculate the speed error, its integral and its derivative
        current_speed = self.vehicle.get_velocity()
        speed_error = self.desired_speed - coord2dist(current_speed)
        self.int_speed_error += speed_error
        d_speed_error = speed_error - self.last_speed_error

        # Add the error to the graphs
        self.graphs_error["speed_error"].append(speed_error)

        self.last_speed_error = speed_error

        # Calculate the throttle and brake using the PID controller 
        throttle = min(max(self.PID_long["Kp"]*speed_error + self.PID_long["Ki"]*self.int_speed_error + self.PID_long["Kd"]*d_speed_error, 0), self.max_throttle)
        brake = min(self.PID_long["Kp"]*speed_error + self.PID_long["Ki"]*self.int_speed_error + self.PID_long["Kd"]*d_speed_error, 0)
        
        return throttle, brake

    def lateral_control(self, lookahead_dist):
        """ 
        Function to calculate the lateral control, i.e. the steer

        :param lookahead_dist: distance to the point to go to
        :return: steer, angle_error (the angle used in the PID controller)
        """


        # Find the closest waypoint to the vehicle
        current_waypoint = self.vehicle.get_world().get_map().get_waypoint(self.vehicle.get_location())
        closest_waypoint = min(self.route, key=lambda x: x[0].transform.location.distance(current_waypoint.transform.location))
        closest_index = self.route.index(closest_waypoint)

        # Find the point to go to waypoint (lookahead waypoint or next_waypoint)
        if closest_index + lookahead_dist < len(self.route):
            next_waypoint = self.route[closest_index + lookahead_dist][0]
        else:
            # If the lookahead waypoint is out of the route, add a new waypoint to the route 
            # Choose the waypoint that is 2 meters ahead of the last waypoint
            next_waypoint = self.route[-1][0]
            yaw = next_waypoint.transform.rotation.yaw*np.pi/180
            forward_vector = carla.Vector3D(x = np.cos(yaw), y = np.sin(yaw))
            next_location = next_waypoint.transform.location + carla.Location(x = forward_vector.x*2, y = forward_vector.y*2)
            next_waypoint = self.vehicle.get_world().get_map().get_waypoint(next_location)
            self.route.append((next_waypoint, 0))

        # Calculate the angle error, its integral and its derivative
        angle_error = find_angle_error_vehicle2nextWaypoint(self.vehicle, next_waypoint)
        self.int_angle_error += angle_error
        d_angle_error = angle_error - self.last_angle_error

        # Calculate the steer using the PID controller
        steer = self.PID_lat["Kp"]*angle_error + self.PID_lat["Ki"]*self.int_angle_error + self.PID_lat["Kd"]*d_angle_error
        
        return steer, angle_error
    
    def run_step_function(self, lookahead_dist, sem_camera_step_save = 0.5, camera_step_save = 0.5, save_data = False, start_time_camera = time.time()):
        """
        The control algorithm and the data saving function

        :param lookahead_dist: distance to the point to go to
        :param sem_camera_step_save: time between two saves of the semantic camera image
        :param camera_step_save: time between two saves of the camera image
        :param save_data: bool to save the data or not
        :param start_time_camera: time when the last image was saved
        :return: bool to continue the control loop or not, start_time_camera

        """
        
        # Calculate the lateral and longitudinal control
        throttle, brake = self.longitudinal_control()
        steer, angle_error = self.lateral_control(lookahead_dist)

        # Add data to the graphs
        self.graphs_error["angle_error"].append(angle_error)
        self.graphs_control["throttle"].append(throttle)
        self.graphs_control["steer"].append(steer)
        self.graphs_control["brake"].append(brake)

        self.last_angle_error = angle_error
        # Apply the control
        self.vehicle.apply_control(carla.VehicleControl(throttle = throttle, steer = steer, brake = brake))

        # Initialize the bool variable to see if the image number should be incremented
        image_number_save = False

        # Save the data
        if bool(self.sensor_data):
            if "collision_sensor" in self.sensors:
                if self.sensor_data["collision"]:
                    print('Collision detected!')
                    return False, None        
            if save_data:
                if "camera" in self.sensors:
                    if time.time() - start_time_camera > camera_step_save:
                        cv2.imwrite('/images/test_detection/town2/Camera' + str(self.image_number+1) + ".png", self.sensor_data['image'])
                        start_time_camera = time.time()
                        image_number_save = True
                if "sem_camera" in self.sensors:
                    if time.time() - start_time_camera > sem_camera_step_save:
                        cv2.imwrite('images/Camera/Sem_camera' + str(self.image_number+1) + ".png", self.sensor_data['sem_image'])
                        start_time_camera = time.time()
                        image_number_save = True

                if image_number_save:
                    # Change image number for saving images
                    self.image_number += 1
                
                image_number_save = False

        return True, start_time_camera
    
    def decision_making_bloc(self, lookahead_dist, start_time = time.time(), dist_error_max = 0.5, time_limit = 250, sem_camera_step_save = 0.5, camera_step_save = 0.5, save_data = False, start_time_camera = time.time()):
        """
        The decision making bloc of the control algorithm that : 
        - Moves the spectator
        - Checks if the vehicle is close enough to the goal
        - Checks if the vehicle is stuck by settign a time limit
        - Adds the data to the graphs
        
        """
        while self.run_step:
            start = time.time()
            self.run_step, start_time_camera = self.run_step_function(lookahead_dist, sem_camera_step_save, camera_step_save, save_data, start_time_camera)

            # Change spectator position to camera position
            self.spectator.set_transform(self.sensors["camera"].get_transform())

            # Check if the vehicle is close enough to the goal
            dist_error = self.goal.distance(self.vehicle.get_location())
            if dist_error < dist_error_max:
                # Stop the vehicle
                self.vehicle.apply_control(carla.VehicleControl(throttle = 0, steer = 0, brake = 1))
                self.time = time.time() - self.time
                self.run_step = True
            
            # Check if the vehicle is stuck by settign a time limit
            if time.time() - start_time > time_limit:
                # Stop the vehicle
                self.vehicle.apply_control(carla.VehicleControl(throttle = 0, steer = 0, brake = 1))
                self.time = time.time() - self.time
                self.run_step = False
            
            # Add the data to the graphs
            self.graphs_error["distance_error"].append(dist_error)
            self.graph_route.append([self.vehicle.get_location().x, self.vehicle.get_location().y])
            self.graphs_values["speed"].append(self.vehicle.get_velocity())
            self.graphs_values["angle"].append(self.vehicle.get_transform().rotation.yaw)

            print("Time for control: ", time.time() - start)

    def lane_detection_bloc(self):
        pass

    def lane_annotation(self, distance_from_vehicle = 30):
        """
        Function to annotate the lanes

        :param distance_from_vehicle: The radius around the vehicle in which the lanes are annotated
        """

        image_number = 0
        while self.run_step:
            print("Lane annotation")
            start = time.time()
            image = self.sensor_data['image']

            # Convert the image to the right format
            img = np.reshape(np.copy(image), (self.sensor_data['image'].shape[0], self.sensor_data['image'].shape[1], 4))

            # get the transformation matrix from the world to the camera
            world_2_camera = np.array(self.sensors["camera"].get_transform().get_inverse_matrix())

            point_image = np.array([[0, 0, -1]])
            for point in self.lane_points:
                x, y = point[:2]
                c = point[2]
                # color = self.colors[int(c % len(self.colors))]
            
                location = carla.Location(x, y, 0.0) 

                # Check if the point is in the radius around the vehicle
                if location.distance(self.vehicle.get_transform().location) < distance_from_vehicle:
                    p = get_image_point(location, self.K, world_2_camera)
                    # Check if the point is in the image
                    if p[1]> self.sensor_data['image'].shape[0]//2:
                        # draw the points if needed
                        # cv2.circle(img, (int(p[0]), int(p[1])), 2, color, -1)
                        point_image = np.append(point_image ,[[p[0], p[1], c]], axis = 0)

            # Delete the first row of the array
            point_image = np.delete(point_image, 0, axis = 0)
            point_image = point_image[np.logical_and(point_image[:, 0] > 0, point_image[:, 1] > 0)]
            point_image = point_image[np.argsort(point_image[:, 2])]
            write_lanes_to_file(point_image, 'detection_voie_Carla/lane_annotation'+ str(image_number) + '.txt')
            cv2.imwrite('detection_voie_Carla/lane_annotation'+ str(image_number) + '.png', img)

            image_number += 1

            cv2.imshow('camera', img)
            cv2.waitKey(1)

            print("Time for lane annotation: ", time.time() - start)

    def TS_detection_bloc(self):

        """
        Function to detect the traffic signs using the traffic signs detection model
        """
        
        with self.detection_graph.as_default():
            while self.run_step:
                start = time.time()
                # Check if the sensors are available
                # If not, initialize the sensor data to zeros
                if "camera" not in self.sensors:
                    self.sensor_data['image'] = np.zeros((self.sensor_data['sem_image'].shape[0], 0, 4), dtype=np.uint8)
                if "sem_camera" not in self.sensors:
                    self.sensor_data['sem_image'] = np.zeros((self.sensor_data['image'].shape[0], 0, 4), dtype=np.uint8)
                sem_image = self.sensor_data['sem_image']
                # Check if the sensors have new data
                image = cv2.cvtColor(self.sensor_data['image'], cv2.COLOR_BGRA2RGB)
                
                # Traffic signs detection 
                detection_image, traffic_sign, _ = TS_detection(image, self.detection_graph, self.category_index, self.sess, 0.99, visualize=True)
                # convert the image to the right format
                self.detection_image = cv2.cvtColor(detection_image, cv2.COLOR_RGB2BGRA)
                self.data_tiled = np.concatenate((self.detection_image, sem_image), axis=1)

                print("Time for detection: ", time.time() - start)
                
    def display_camera(self, display_cameras = True, save_video = False):
        """
        Function to display the camera images and save the video if needed

        :param display_cameras: bool to display the camera images or not
        :param save_video: bool to save the video or not
        """
        if save_video:
            self.video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (IM_HEIGHT, IM_WIDTH))
        # Dispaly with imshow
        while self.run_step:
            if display_cameras:
                if np.all(self.data_tiled[0] != None):
                    cv2.imshow('All cameras', self.data_tiled)
                    cv2.waitKey(1)
                    if save_video:
                        self.video.write(cv2.cvtColor(self.data_tiled, cv2.COLOR_RGBA2RGB))

        if save_video:
            self.video.release()
        
    def run_navigation(self, lookahead_dist, start_time = time.time(), display_cameras = False, dist_error_max = 0.5, time_limit = 250, sem_camera_step_save = 0.5, camera_step_save = 0.5, save_data = False, start_time_camera = time.time(), save_video = False):
        """
        Function to run the control bloc and the perception blocs in parallel

        :param lookahead_dist: distance to the point to go to
        :param start_time: time when the control loop started
        :param display_cameras: bool to display the camera images or not
        :param dist_error_max: max distance error to the goal
        :param time_limit: max time for the control loop
        :param sem_camera_step_save: time between two saves of the semantic camera image
        :param camera_step_save: time between two saves of the camera image
        :param save_data: bool to save the data or not
        :param start_time_camera: time when the last image was saved

        :return: bool to continue the control loop or not
        """
        # th1 = th.Thread(target=self.TS_detection_bloc)
        th2 = th.Thread(target=self.decision_making_bloc, args=(lookahead_dist, start_time, dist_error_max, time_limit, sem_camera_step_save, camera_step_save, save_data, start_time_camera))
        # th3 = th.Thread(target=self.display_camera, args=(display_cameras, save_video))
        th4 = th.Thread(target=self.lane_annotation)

        # th1.start()
        th2.start()
        # th3.start()
        th4.start()

        # th1.join()
        th2.join()
        # th3.join()
        th4.join()

        self.time = time.time() - self.time
        return False
    
    def draw_graph_error(self, graph_name):
        """
        draw the graph of the error and calculate the mean and max error
        
        :param graph_name: name of the graph to draw (distance_error, speed_error, angle_error)

        :return: error_mean, error_max
        """
        if graph_name == "speed_error":
            self.graphs_error[graph_name] = np.array(self.graphs_error[graph_name]) * 3.6

        timing = np.linspace(0, time.time() - self.time, len(self.graphs_error[graph_name]))
        plt.plot(timing, self.graphs_error[graph_name])
        plt.title(graph_name.split("_")[0] + " " + graph_name.split("_")[1])
        plt.ylabel("Erreur en vitesse (km/h)")
        plt.xlabel("Time (s)")
        plt.grid()
        plt.show()

        error_mean = np.mean(self.graphs_error[graph_name])
        error_max = np.max(self.graphs_error[graph_name])
 
        print(graph_name, "Error mean: ", error_mean)
        print(graph_name, "Error max: ", error_max)

        return error_mean, error_max
    

    def draw_graph_control(self, graph_name):
        """
        Draw the graph of the control

        :param graph_name: name of the graph to draw (throttle, steer, brake)
        """
        timing = np.linspace(0, self.time, len(self.graphs_error[graph_name]))
        plt.plot(timing, self.graphs_control[graph_name])
        plt.title(graph_name)
        plt.ylabel(graph_name)
        plt.xlabel("Time (s)")
        plt.grid()
        plt.show()

    def draw_graph_values(self, graph_name):
        """
        Draw the graph of the speed or the yaw angle of the vehicle

        :param graph_name: name of the graph to draw (speed, angle)
        """
        timing = np.linspace(0, self.time, len(self.graphs_error[graph_name]))
        plt.plot(timing, self.graphs_values[graph_name])
        plt.title(graph_name)
        plt.ylabel(graph_name)
        plt.xlabel("Time (s)")
        plt.grid()
        plt.show()

    def draw_route(self):

        """
        Draw the route and the route followed

        :return: mean_dist_error, max_dist_error
        """

        x = [x[0].transform.location.x for x in self.route]
        y = [x[0].transform.location.y for x in self.route]

        # Draw the route
        for i in range(len(self.route)):
            if not(self.route[i][1] == 0):
                plt.arrow(x[i], y[i], 5*np.cos(self.route[i][0].transform.rotation.yaw*np.pi/180), 5*np.sin(self.route[i][0].transform.rotation.yaw*np.pi/180), head_width=1, head_length=1, fc='k', ec='k')
                plt.scatter(x[i], y[i], c='k', s=3)
        
        plt.scatter(x[i-1], y[i-1], c='#000000', s=3, label = 'Ensemble des points à suivre')

        # Draw the route followed
        x_followed = [xf[0] for xf in self.graph_route]
        y_followed = [yf[1] for yf in self.graph_route]
        plt.title('Route')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.plot(x_followed, y_followed, label = 'Chemin parcouru')
        plt.legend()
        plt.grid()
        plt.show()
            
        # Calculate the distance error between the route and the route followed
        x = np.array(x)
        y = np.array(y)

        dist_error = []
        for i in range(len(x)):
            index_closest = np.argmin((x[i]- x_followed)**2 + (y[i] - y_followed)**2)
            x_followed_closest, y_followed_closest = x_followed[index_closest], y_followed[index_closest]
            dist_error.append(np.sqrt((x[i] - x_followed_closest)**2 + (y[i] - y_followed_closest)**2))

        mean_dist_error = np.mean(dist_error)
        max_dist_error = np.max(dist_error)

        print("Mean distance error: ", mean_dist_error)
        print("Max distance error: ", max_dist_error)

        return mean_dist_error, max_dist_error