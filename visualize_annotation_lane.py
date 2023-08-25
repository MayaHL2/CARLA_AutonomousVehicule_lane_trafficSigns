"""
This file contains the functions used to visualize the annotation of the lanes in an image.
It is used when creating the dataset to test if the annotation is correct.
"""

import cv2
import numpy as np

def read_points_from_txt(txt_file):
    """
    Function to read the points from the txt file

    :param txt_file: path to the txt file containing the points

    :return: points: points read from the txt file
    """
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    points = []
    for line in lines:
        line_lane = np.float32(line.strip().split())
        line_lane = np.reshape(line_lane, (-1, 2))
        points.append(line_lane)

    return np.vstack(points)


def draw_points_on_image(image, points, colors):
    """
    Function to draw the points on the image using cv2.circle

    :param image: image to draw the points on
    :param points: points to draw on the image
    :param colors: colors used to define the lane number (this hasn't been used in this project)
    """
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), 3, colors[0], -1)


# Directory path and color definitions
dataset_dir = 'detection_voie_Carla/town1'
image_prefix = ''
txt_extension = '.lines.txt'
image_extension = '.jpg'
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 128, 128), (192, 192, 192), (255, 165, 0), (210, 105, 30), (139, 69, 19), (160, 82, 45), (205, 133, 63), (244, 164, 96), (255, 69, 0), (0, 100, 0), (0, 128, 128), (0, 191, 255), (100, 149, 237), (72, 61, 139), (128, 0, 0), (139, 0, 0), (205, 92, 92), (255, 105, 180), (218, 112, 214), (186, 85, 211), (255, 192, 203), (255, 218, 185)]

# Process each image and its corresponding txt file
for idx in range(1, 100):  # Change the range based on the number of images you have
    image_path = f"{dataset_dir}/{image_prefix}{idx:d}{image_extension}"
    txt_path = f"{dataset_dir}/{image_prefix}{idx:d}{txt_extension}"

    # Read the image and the points from the txt file
    image = cv2.imread(image_path)
    lane_points = read_points_from_txt(txt_path)


    # print(lane_points[np.logical_and(lane_points[:, 0] > 0, lane_points[:, 1] > 0)].shape)
    # print(lane_points[np.logical_and(lane_points[:, 0] > 0, lane_points[:, 1] > 0)])

    # Draw the points on the image
    draw_points_on_image(image, lane_points, colors)



    # Display or save the image with the points
    # For displaying the image, use the following line:
    cv2.imshow(f"Image with Lanes", image)
    cv2.waitKey(100)
cv2.destroyAllWindows()
# For saving the image with the points, use the following line:
# cv2.imwrite(f"{dataset_dir}/output_{image_prefix}{idx:03d}{image_extension}", image)
