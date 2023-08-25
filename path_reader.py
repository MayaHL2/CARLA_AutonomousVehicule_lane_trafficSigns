"""
Create a text file containing the path to all images in a directory.
"""
import os

# Path to the directory containing the data
root_path = 'detection_voie_Carla'

# Write path to all images in a text file
with open('dataset_lane_carla.txt', 'w') as f:
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            if name.endswith('.png'):
                f.write(os.path.join(path, name) + '\n')


