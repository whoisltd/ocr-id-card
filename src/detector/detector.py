from charset_normalizer import detect
import numpy as np
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from src.detector.utils.image_utils import non_max_suppression_fast


class Detector(object):
    def __init__(self, path_to_model, path_to_labels, nms_threshold=0.3, score_threshold=0.3):
        self.path_to_model = path_to_model
        self.path_to_labels = path_to_labels
        self.category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        # load model
        self.interpreter, self.detect_fn = self.load_model()
        self.detection_scores = None
        self.detection_boxes = None
        self.detection_classes = None

    def load_model(self):
        # Load the saved model and allocate tensors.
        print('Loading model...', end='')
        start_time = time.time()
        model = tf.saved_model.load(self.path_to_model)
        detect_fn = model.signatures['serving_default']
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {:.2f}s'.format(elapsed_time))

        return model, detect_fn

    def predict(self, img):
        # original = img
        # img = np.array(img)
        input_tensor = tf.convert_to_tensor(img)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.detect_fn(input_tensor)

        # Retrieve detection results
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        self.detection_boxes = detections['detection_boxes']  # Bounding box coordinates of detected objects
        self.detection_classes = detections['detection_classes']  # Class index of detected objects
        self.detection_scores = detections['detection_scores']  # Confidence of detected objects

        mask = np.array(self.detection_scores) > self.score_threshold
        self.detection_boxes = np.array(self.detection_boxes)[mask]
        self.detection_classes = np.array(self.detection_classes)[mask]

        # Convert coordinate to original coordinate
        h, w, _ = img.shape
        self.detection_boxes[:, 0] = self.detection_boxes[:, 0] * h
        self.detection_boxes[:, 1] = self.detection_boxes[:, 1] * w
        self.detection_boxes[:, 2] = self.detection_boxes[:, 2] * h
        self.detection_boxes[:, 3] = self.detection_boxes[:, 3] * w

        # Apply non-max suppression
        self.detection_boxes, self.detection_classes = non_max_suppression_fast(boxes=self.detection_boxes,
                                                                                labels=self.detection_classes,
                                                                                overlapThresh=self.nms_threshold)
        return self.detection_boxes, np.array(self.detection_classes).astype("int"), self.category_index
