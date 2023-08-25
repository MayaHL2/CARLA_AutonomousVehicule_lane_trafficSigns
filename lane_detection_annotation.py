import numpy as np
import carla
import networkx as nx 

FOV = 90.0
IM_WIDTH = 512
IM_HEIGHT = 512
 

def build_rotation_matrix(yaw, pitch, roll):
    """
    Build the transformation/rotation matrix using the yaw, pitch and roll angles

    :param yaw: yaw angle in degrees
    :param pitch: pitch angle in degrees
    :param roll: roll angle in degrees

    :return: rotation matrix
    """


    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)

    Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))

    return R


def build_projection_matrix(w = IM_WIDTH, h = IM_HEIGHT, fov = FOV):
    """
    Build the projection matrix using the width/height of the camera and its fov
    The projection matrix is used to transform the 3D coordinates to 2D
    
    :param w: width of the camera
    :param h: height of the camera
    :param fov: field of view of the camera

    :return: projection matrix
    """
    
    w = int(w)
    h = int(h)
    fov = float(fov)

    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    """
    Get the 2D projection of a 3D coordinate using the camera matrix and the world to camera matrix

    :param loc: 3D coordinate to project, it is a carla.Position object
    :param K: camera matrix (intrinsics characteristics)
    :param w2c: world to camera matrix

    :return: 2D projection of the 3D coordinate
    """

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # Change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # Remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # Project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def world2camera(camera):
    """
    Get the world to camera matrix from a camera object 
    Get the transformation matrix from the world to the camera

    :param camera: camera object (used to get the transformation matrix from the camera to the world)

    :return: world to camera matrix
    """

    # Get the transformation matrix from the camera to the world
    camera_transform = camera.get_transform()

    # Invert camera transform to get the world to camera matrix
    return np.array(camera_transform.get_inverse_matrix())


def generate_lanes(waypoints):
    """
    For a given list of waypoints, generate left and right lanes
    The return is two arrays of carla.Transform objects, one for the left lane and one for the right lane

    :param waypoints: list of waypoints

    :return: left and right lanes 
    """

    i = 0

    locations = []
    rotations = []
    # Get the location and rotation of each waypoint
    for waypoint in waypoints:
        location = waypoint.transform.location
        rotation = waypoint.transform.rotation

        locations.append([location.x, location.y, location.z])
        rotations.append([rotation.yaw])

    # find the left and right lanes of each waypoint
    left_lanes = []
    right_lanes = []
    for i in range(0, len(locations)):
        # Get the lane width of the waypoint
        lane_width = waypoints[i].lane_width * 0.5

        # Use the lane width and the rotation of the waypoint to find the left and right lanes
        left_lane = lane_width*np.array([-np.sin(np.deg2rad(rotations[i][0])), np.cos(np.deg2rad(rotations[i][0])), 0]) + locations[i]
        right_lane = lane_width*np.array([np.sin(np.deg2rad(rotations[i][0])), -np.cos(np.deg2rad(rotations[i][0])), 0]) + locations[i]
        
        # Check if the waypoint has a lane, i.e. if it is not in an intersection
        if waypoints[i].get_left_lane() != None:
            
            # Add the left and right lanes to arrays
            left_lanes.append(carla.Transform(carla.Location(x=left_lane[0], y=left_lane[1], z=left_lane[2]), carla.Rotation(pitch=0, yaw=rotations[i][0], roll=0)))
            right_lanes.append(carla.Transform(carla.Location(x=right_lane[0], y=right_lane[1], z=right_lane[2]), carla.Rotation(pitch=0, yaw=rotations[i][0], roll=0)))


    left_lanes = np.array(left_lanes)[np.array(left_lanes) != None]
    right_lanes = np.array(right_lanes)[np.array(right_lanes) != None]

    return left_lanes, right_lanes

def waypointsLane2array(left, right):
    """
    Convert the left and right lanes to an array of points
    The points are in the format [x, y, theta], where x and y are the coordinates of the point and theta is the rotation of the lane at that point
    Remove the duplicate points

    :param left: left lanes
    :param right: right lanes

    :return: array of points
    """

    locations = []

    for lane in np.hstack([left, right]):
        locations.append([lane.location.x, lane.location.y])


    # Creating plot
    x_left = []
    y_left = []
    x_right = []
    y_right = []
    rot = []
    for i in range(len(left)):
        x_left.append(left[i].location.x)
        y_left.append(left[i].location.y)
        rot.append(left[i].rotation.yaw)

    x_left = np.array(x_left)
    y_left = np.array(y_left)

    left_location = np.hstack([x_left.reshape(-1, 1), y_left.reshape(-1, 1)])

    for i in range(len(right)):
        x_right.append(right[i].location.x)
        y_right.append(right[i].location.y)
    
    x_right = np.array(x_right)
    y_right = np.array(y_right)

    rotation = np.array(rot)

    right_location = np.hstack([x_right.reshape(-1, 1), y_right.reshape(-1, 1)])

    locations = np.vstack([left_location, right_location])
    locations_index = np.unique(floor_round(locations, decimal_places=1), axis=0, return_index=True)[1]

    duplicate_index = np.setdiff1d(range(len(locations)), locations_index)
    
    locations = np.delete(locations, duplicate_index, axis=0)
    rotation = np.delete(np.hstack([rotation, rotation]), duplicate_index, axis=0)

    points = np.hstack([locations, rotation.reshape(-1, 1)])

    return points

def find_closest_point_in_direction(p, points, max_distance=10, max_angle=8):
    """
    Find the closest point to a given point in its direction
    The direction is given by the angle (yaw) of the point
    This is equivalent to find the closest point in a cone in front of the given point
    Which is used to find the next point in a lane

    :param p: point to find the closest point in its direction
    :param points: array of points
    :param max_distance: maximum distance to consider a point as a neighbor
    :param max_angle: maximum angle to consider a point as a neighbor

    :return: closest point and its index
    """


    points_close = np.array(points)

    distance = np.linalg.norm(points_close[:, :2] - p[:2], axis=1)
    points_close = points_close[distance < max_distance]
    points_close = points_close[np.linalg.norm(points_close - p, axis=1) > 0.01]
    
    if len(points_close) == 0:
        return None, None
    else:
        angle_dir = np.arctan2( points_close[:, 1] - p[1], points_close[:, 0] - p[0] )
        angle_dir = angle_dir - np.deg2rad(p[2])
        angle_dir = np.abs(np.arctan2(np.sin(angle_dir), np.cos(angle_dir)))

        points_close = points_close[angle_dir < np.deg2rad(max_angle)]        
    

        if len(points_close) == 0:
            return None, None
        
        else:
            closest_point = points_close[np.argmin(np.linalg.norm(points_close[:, :2] - p[:2], axis=1))]
            index_closest_point = np.where(np.all(points == closest_point, axis = 1))[0][0]
            return closest_point, index_closest_point
    

def classify_lanes(points, max_distance=10, max_angle=8):
    """
    Classify the points in lanes
    For each point, we find the closest point in its direction which is considered the next point in the lane
    Then, we find the closest point in the opposite direction which is considered the previous point in the lane
    We consider the previous and next points as neighbors of the point and create a graph
    Finally, we find the connected components of the graph which are the lanes, i.e. find the connected subgraphs

    :param points: array of points
    :param max_distance: maximum distance to consider a point as a neighbor (used in find_closest_point_in_direction)
    :param max_angle: maximum angle to consider a point as a neighbor (used in find_closest_point_in_direction)

    :return: dictionary of lanes, where the key is the lane number and the value is an array of points
    """
    neighbor_index = []

    for i in range(len(points)):

        pt = points[i]
        # Find the next_point in the direction of the point
        _, index_next = find_closest_point_in_direction(pt, points, max_distance, max_angle)

        pts = np.hstack((-points[:, :2], points[:, 2:]))

        pt = pt.reshape(1, -1) 
        pt = np.hstack((-pt[:, :2], pt[:, 2:]))
        pt = np.squeeze(pt)
        # Find the previous_point in the opposite direction of the point
        # This is done by finding the next_point in the opposite direction of the point
        _, index_prev = find_closest_point_in_direction(pt, pts)

        neighbor_index.append([index_prev, index_next])


    # Array of the indexes of the neighbors of each point ordered by the point index
    neighbor_index = np.array(neighbor_index)

    # Create a graph where the nodes are the points and the edges are the neighbors
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))

    # Add the edges to the graph using the neighbor_index
    for i in range(len(points)):
        if neighbor_index[i, 0] != None:
            G.add_edge(i, neighbor_index[i, 0])
        if neighbor_index[i, 1] != None:
            G.add_edge(i, neighbor_index[i, 1])

    # Find the connected components of the graph which are the lanes
    subgraphs = (G.subgraph(c) for c in nx.connected_components(G))
    subgraphs = list(subgraphs)

    lanes = dict()

    l = 0
    # Write the lanes in a dictionary
    for subgraph in subgraphs:
        temp = list(subgraph.nodes)
        lanes[l] = points[temp]

        l += 1

    return lanes


def lane2points_class(lanes):
    """
    Convert the dictionary of lanes to an array of points with the class label [x, y, class_label]

    :param lanes: dictionary of lanes

    :return: array of points with the class label
    """

    # Count the total number of points
    num_points = sum(len(points) for points in lanes.values())

    # Create an array with shape (n, 3)
    points_array = [[None] * 3 for _ in range(num_points)]

    # Iterate over the classes and points to populate the array
    index = 0
    for class_label, points in lanes.items():
        for point in points:
            x, y, theta = point
            points_array[index] = [x, y, class_label]
            index += 1

    return np.array(points_array)

def write_lanes_to_file(points_class, filename):
    """
    write the points with the class label to a text file
    Each class (lane) is written in a new line

    :param points_class: array of points with the class label
    :param filename: name of the file to write the points

    :return: None
    """


    # Create a string representation of the points in the desired format
    lines = []
    current_class = None
    for point in points_class:
        x, y, class_label = point
        if class_label != current_class:
            current_class = class_label
            if lines:
                lines[-1] = lines[-1].strip()  # Remove trailing space from the previous line
                lines.append('\n')  # Add a new line before starting the next class
        lines.append(f'{x} {y} ')
    lines[-1] = lines[-1].strip()  # Remove trailing space from the last line

    # Write the lines to a text file
    with open(filename, 'w') as file:
        file.writelines(lines)
        

def floor_round(value, decimal_places=1):
    """
    Round a number to a given number of decimal places

    :param value: number to round
    :param decimal_places: number of decimal places to round

    :return: rounded number
    """
    scale = 10 ** decimal_places
    return np.floor(value * scale) / scale

if __name__ == "__main__":

    
    import matplotlib.pyplot as plt    


    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # world = client.get_world()
    world = client.load_world('Town10HD')
    map = world.get_map()

    waypoints = map.generate_waypoints(5)

    left, right = generate_lanes(waypoints)

    points = waypointsLane2array(left, right)


    colors = ["r", "g", "b", "c", "m", "y", "k", "w", "maroon", "darkred", "firebrick", "orangered", "gold", "yellow", "olive", "lime", "lawngreen", "greenyellow", "chartreuse", "aquamarine", "mediumturquoise", "cyan", "deepskyblue", "blueviolet", "magenta", "orchid", "plum", "thistle", "purple", "mediumpurple", "darkmagenta", "indigo", "hotpink", "coral", "darkorange", "sandybrown", "tan"]
    np.random.shuffle(colors)


    lanes = classify_lanes(points)
    points_class = lane2points_class(lanes)

    for lane_number, lane_points in lanes.items():
        print(lane_number, len(colors))
        if lane_number > 33:
            break
        plt.scatter(lane_points[:, 0], lane_points[:, 1], color=colors[lane_number], s = 4)

    plt.grid()
    plt.show()