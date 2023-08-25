import json
import csv
import numpy as np
import cv2

import cv2

def draw_bounding_box_cv2(image, bbox):

    x_min, y_min, x_max, y_max = bbox

    # Draw the bounding box rectangle on the image
    return cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

def extract_traffic_signs(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    return image[y_min:y_max, x_min:x_max]

def write_to_csv(file_path, data):

    with open(file_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for row in data:
            writer.writerow(row)

def update_label_csv(input_csv_path, output_csv_path, target_numbers, replacement_numbers):

        with open(input_csv_path, 'r', newline='') as csvfile_in:
            with open(output_csv_path, 'w', newline='') as csvfile_out:
                reader = csv.reader(csvfile_in)
                writer = csv.writer(csvfile_out)

                # Process each row in the CSV file
                for row in reader:
                    # Check if the row has enough elements (columns) before updating the 7th column
                    if len(row) >= 7:
                        try:
                            # Try to convert the value in the 7th column to an integer
                            label = int(row[6])

                            # Check if the value matches any target number in the list and update it with the corresponding replacement number
                            if label in target_numbers:
                                index = target_numbers.index(label)
                                row[6] = str(replacement_numbers[index])
                        except ValueError:
                            pass

                    # Write the updated row to the output CSV file
                    writer.writerow(row)

datadir = "data_100k/"

filedir = datadir + "/annotations.json"
ids_test = open(datadir + "/test/ids.txt").read().splitlines()
ids_train = open(datadir + "/train/ids.txt").read().splitlines()

annos = json.loads(open(filedir).read())

i = 0
data_labels = []

labels = np.array([20, 30, 50, 60, 70, 80, 90, 100, 120])

x = []
for key in annos['imgs']:

    image_path = 'data_100k/' + annos['imgs'][key]['path']
    image = cv2.imread(image_path)
    
    traffic_signs = []
    for obj in annos['imgs'][key]['objects']:
        bbox = obj['bbox']
        bbox = np.int16([max(0, min(image.shape[1], bbox['xmin'])), max(0, min(image.shape[1],bbox['ymin'])), max(0, min(image.shape[1],bbox['xmax'])), max(0, min(image.shape[1],bbox['ymax']))])
        # image = draw_bounding_box_cv2(image, np.int16(bbox))
        ts = extract_traffic_signs(image, bbox)
        traffic_signs.append(ts)

        if obj['category'][:2] == 'pl' and obj['category'][-1] != '5' and obj['category'][-2] != '4' and obj['category'][-1] != '3' and obj['category'][-1] != '3' and obj['category'][-1] != '4' and obj['category'][-2:] != '10' and obj['category'][-3:] != '110' and obj['category'] != 'pl0':
            cv2.imwrite('images/100k/images/' + str(i) + '.png', ts)
            label = np.where(labels == int(obj['category'][2:]))[0][0]
            data_labels.append([0,0,0,0,0,0,label,'images/' + str(i)+ '.png'])

            i += 1


write_to_csv('images/100k/labels.csv', data_labels)


# update_label_csv('images/100k/labels.csv', 'images/100k/labels_new.csv', [], [])