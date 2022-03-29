import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder
import time


class Detector(object):
    def __init__(self, path_config, path_ckpt, path_to_labels):
        self.path_config = path_config
        self.path_ckpt = path_ckpt
        self.label_path = path_to_labels

        self.category_index = label_map_util.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)
        self.detection_model = self.load_model()
        self.detection_scores = None
        self.detection_boxes = None
        self.detection_classes = None

    def detect_fn(self, image):
        """Detect objects in image."""
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def load_model(self):
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(self.path_config)
        model_config = configs['model']
        detection_model = model_builder.build(
            model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(self.path_ckpt).expect_partial()

        return detection_model

    def predict(self, image):
        original_img = np.copy(image)

        image = np.asarray(image)

        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        # num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)

        self.detection_scores = detections['detection_scores']
        self.detection_classes = detections['detection_classes']
        self.detection_boxes = detections['detection_boxes']

        # draw bounding boxes and labels
        image, coordinate_dict = self.draw(image)

        return image, original_img, coordinate_dict

    def draw(self, img):
        coordinate_dict = dict()
        height, width, _ = img.shape
        li = []

        for i, score in enumerate(self.detection_scores):
            if score < 0.3:
                continue

            self.detection_classes[i] += 1
            # if background, ignore
            if self.detection_classes[i] == 0:
                continue

            label = str(self.category_index[self.detection_classes[i]]['name'])
            ymin, xmin, ymax, xmax = self.detection_boxes[i]
            real_xmin, real_ymin, real_xmax, real_ymax = int(xmin * width), int(ymin * height), int(xmax * width), int(
                ymax * height)

            curr = real_xmax * real_ymax - real_ymin * real_xmin
            status = check_overlap(curr, li)
            if status == 1:
                continue

            li.append(real_xmax * real_ymax - real_ymin * real_xmin)
            # check overlap bounding boxes

            cv2.rectangle(img, (real_xmin, real_ymin),
                          (real_xmax, real_ymax), (0, 255, 0), 2)
            cv2.putText(img, label, (real_xmin, real_ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale=0.5)
            coordinate_dict[label] = (
                real_xmin, real_ymin, real_xmax, real_ymax)

        return img, coordinate_dict


def check_overlap(curr, li):
    for va in li:
        # overlap
        if abs(va - curr) < 1000:
            return 1
    return 0


image = cv2.imread('/home/whoisltd/Desktop/as6.jpg')
detector = Detector('/home/whoisltd/detect/training/exported-models/my_model/pipeline.config', '/home/whoisltd/detect/training/exported-models/my_model/checkpoint/ckpt-0', '/home/whoisltd/detect/tf_record_text/label_map.pbxt')
# detector = Detector('/home/whoisltd/detect/training/exported-models-corner/my_model/pipeline.config',
#                     '/home/whoisltd/detect/training/exported-models-corner/my_model/checkpoint/ckpt-0', '/home/whoisltd/detect/tf_record/label_map.pbxt')
start = time.time()
image, original_image, coordinate_dict = detector.predict(image)


def get_center_point(coordinate_dict):
    di = dict()

    for key in coordinate_dict.keys():
        xmin, ymin, xmax, ymax = coordinate_dict[key]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        di[key] = (x_center, y_center)

    return di


def find_miss_corner(coordinate_dict):
    dict_corner = {'top_left': 0, 'top_right': 1,
                   'bottom_left': 2, 'bottom_right': 3}
    for key in dict_corner.keys():
        if key not in coordinate_dict.keys():
            return dict_corner[key]

# def find_miss_corner(coordinate_dict):
#     position_name = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
#     position_index = np.array([0, 0, 0, 0])

#     for name in coordinate_dict.keys():
#         if name in position_name:
#             position_index[position_name.index(name)] = 1

#     index = np.argmin(position_index)

#     return index


def calculate_missed_coord_corner(coordinate_dict):
    thresh = 0

    index = find_miss_corner(coordinate_dict)

    # calculate missed corner coordinate
    # case 1: missed corner is "top_left"
    if index == 0:
        midpoint = np.add(
            coordinate_dict['top_right'], coordinate_dict['bottom_left']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_right'][0] - thresh
        coordinate_dict['top_left'] = (x, y)
    elif index == 1:  # "top_right"
        midpoint = np.add(
            coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['bottom_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['bottom_left'][0] - thresh
        coordinate_dict['top_right'] = (x, y)
    elif index == 2:  # "bottom_left"
        midpoint = np.add(
            coordinate_dict['top_left'], coordinate_dict['bottom_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_right'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_right'][0] - thresh
        coordinate_dict['bottom_left'] = (x, y)
    elif index == 3:  # "bottom_right"
        midpoint = np.add(
            coordinate_dict['bottom_left'], coordinate_dict['top_right']) / 2
        y = 2 * midpoint[1] - coordinate_dict['top_left'][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict['top_left'][0] - thresh
        coordinate_dict['bottom_right'] = (x, y)

    return coordinate_dict


def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))

    return dst


def align_image(image, coordinate_dict):
    if len(coordinate_dict) < 3:
        raise ValueError('Please try again')

    # convert (xmin, ymin, xmax, ymax) to (x_center, y_center)
    coordinate_dict = get_center_point(coordinate_dict)

    if len(coordinate_dict) == 3:
        coordinate_dict = calculate_missed_coord_corner(coordinate_dict)

    top_left_point = coordinate_dict['top_left']
    top_right_point = coordinate_dict['top_right']
    bottom_right_point = coordinate_dict['bottom_right']
    bottom_left_point = coordinate_dict['bottom_left']

    source_points = np.float32(
        [top_left_point, top_right_point, bottom_right_point, bottom_left_point])

    # transform image and crop
    crop = perspective_transform(image, source_points)

    return crop


def non_max_suppression_fast(boxes, labels, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    #
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    final_labels = [labels[idx] for idx in pick]
    final_boxes = boxes[pick].astype("int")
    return final_boxes, final_labels

# coord = get_center_point(coordinate_dict)
# coord1 = calculate_missed_coord_corner(coord)
# print(coord)
# crop = align_image(image, coordinate_dict)


end = time.time()
print("Estimated time: ", end - start)
cv2.imwrite('corner_test.png', image)
cv2.imshow('test', image)
cv2.waitKey(0)
