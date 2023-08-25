"""
If we want to train/test the model CondLaneNet for lane detection on the dataset Carla, we need to resize the images
and modify the lables so that they are compatible with the resized image.

This script is used to resize the images and modify the labels.
"""


import cv2
import numpy as np
import os
import re

folder_path = 'detection_voie_Carla\\town10'

# List all files in the folder
files = os.listdir(folder_path)

# Filter only image files
image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]


# Loop through the image files and read them using cv2
for image_file in image_files:
    image = cv2.imread(os.path.join(folder_path, image_file))

    # The resulting image needs to be of size 590x1640
    # We crop the image with 164 pixels on the top and bottom
    # new size: 184x512
    # This new size can easily be resized using cv2.resize without distortion to 590x1640
    image = image[164: 512- 164, :, :]

    # read file detection_voie_Carla\town1\lane_annotation0.txt
    # Open the file in read mode
    number = re.findall(r'\d+', image_file)[0]
    with open(os.path.join(folder_path, 'lane_annotation' + number + '.txt'), 'r') as file:
        content = file.read()

    lanes = []
    content = content.split('\n')
    for lane in content:
        lane = lane.split(' ')
        lane = np.array([float(x) for x in lane])
        lane = lane.reshape(-1, 2)
        lanes.append(lane)

        for i in range(0, len(lane)):
            lane[i, 0]  = int(lane[i, 0])
            lane[i, 1]  = int(lane[i, 1] - 164)


    # Open the file in write mode
    with open(os.path.join(folder_path + '_reshape', number + '.lines.txt'), 'w') as file:
        for lane in lanes:
            lane = lane.reshape(-1)
            line = ' '.join(map(str, lane))  # Join array elements with spaces
            file.write(line + '\n')  # Write the line to the file, followed by a newline character

    # save the image
    cv2.imwrite(os.path.join(folder_path + '_reshape', number + '.jpg'), image)