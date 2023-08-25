"""
This file contains the functions used to detect traffic signs in an image.
The functions are used in the navigation algorithm to detect traffic signs
"""


import numpy as np
import os
import tensorflow as tf

import time

import object_detection.utils.visualization_utils as vis_util
import object_detection.utils.label_map_util as label_map_util

import cv2

PATH_TO_LABELS = os.path.join('gtsdb_data', 'gtsdb_label_map.pbtxt')
# The possible models can be found in the models folder, they were installed from this link:
# https://github.com/aarcosg/traffic-sign-detection
# MODEL_NAME = 'ssd_inception_v2'
MODEL_NAME = 'ssd_mobilenet_v1'


def initialize_model(model_name = MODEL_NAME, path_to_labels = PATH_TO_LABELS):
    """
    Function to initialize the model and the label map

    :param model_name: name of the model to use (ssd_mobilenet_v1 or ssd_inception_v2 or else)
    :param path_to_labels: path to the label map

    :return: detection_graph: the graph of the model
    """
    MODEL_PATH = os.path.join('models', model_name)
    PATH_TO_CKPT = os.path.join(MODEL_PATH,'inference_graph/frozen_inference_graph.pb')

    NUM_CLASSES = 3

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return detection_graph, category_index

def TS_detection(image, detection_graph, category_index, sess, score_threshold = 0.8, visualize = True):
    """
    Function to detect traffic signs in an image

    :param image: image to detect traffic signs in
    :param detection_graph: graph of the model
    :param category_index: index of the categories
    :param sess: session of the model
    :param score_threshold: threshold of the score of the traffic sign
    :param visualize: boolean to visualize the traffic sign in the image

    :return: image_np: image, image of the traffic sign, boxes: boxes of the traffic sign
    """
    image_np = np.copy(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    if visualize:
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=6)
    
    if scores[0][0] >= score_threshold:
        ymin, xmin, ymax, xmax = boxes[0][0]  # Extract ymin, xmin, ymax, xmax from the box
        h, w, _ = image_np.shape    # Get the height and width of the image
        # Convert the normalized coordinates to pixel values
        left = int(xmin * w)
        right = int(xmax * w)
        top = int(ymin * h)
        bottom = int(ymax * h)

        # Extract detected traffic sign
    
        traffic_sign = image_np[top:bottom, left:right]
    else :
        traffic_sign = None
    
    # return boxes that have a score higher than the threshold
    return image_np, traffic_sign, boxes[scores > score_threshold]

if __name__ == "__main__":
    
    from imutils import paths

    imagePaths = list(paths.list_images("images\EVO\\rainy_test"))

    detection_graph, category_index = initialize_model()
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            for (i, imagePath) in enumerate(imagePaths):
                print(i)
                # image_np = np.array(cv2.cvtColor(cv2.imread('images/test_images/Town1/000000.png'), cv2.COLOR_BGR2RGB))
                image = cv2.imread(imagePath)
                if np.all(image == None):
                    # print('Image not found!')
                    continue
                else : 
                    image_np = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    start = time.time()
                    image_detection, traffic_sign, _ = TS_detection(image_np, detection_graph, category_index, sess, visualize=True)
                    # cv2.imshow("image", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                    cv2.imwrite("temp\\" + str(i) + ".png", cv2.cvtColor(image_detection, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(2)
                    print((time.time()-start)*1000, "ms")
    
    cv2.destroyAllWindows()