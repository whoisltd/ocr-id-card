import numpy as np
import cv2

def get_center_point(coordinate_dict):
    points = dict()
    for key in coordinate_dict.keys():
        xmin, ymin, xmax, ymax = coordinate_dict[key]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        points[key] = (x_center, y_center)
    return points

def find_miss_corner(coordinate_dict):
    dict_corner = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    for key in dict_corner:
        if key not in coordinate_dict.keys():
            return key

def calculate_missed_coord_corner(coordinate_dict):
    #calulate a coord corner of a rectangle 
    def calculate_coord_by_mid_point(coor1, coord2, coord3):
        midpoint = np.add(coordinate_dict[coor1], coordinate_dict[coord2]) / 2
        y = 2 * midpoint[1] - coordinate_dict[coord3][1]
        x = 2 * midpoint[0] - coordinate_dict[coord3][0]
        return (x, y)
    # calculate missed corner coordinate
    corner = find_miss_corner(coordinate_dict)
    if corner == 'top_left':
        coordinate_dict['top_left'] = calculate_coord_by_mid_point('top_right', 
        'bottom_left', 'bottom_right')
    elif corner == 'top_right':
        coordinate_dict['top_right'] = calculate_coord_by_mid_point('top_left', 
        'bottom_right', 'bottom_left')
    elif corner == 'bottom_left':
        coordinate_dict['bottom_left'] = calculate_coord_by_mid_point('top_left', 
        'bottom_right', 'top_right')
    elif corner == 'bottom_right':
        coordinate_dict['bottom_right'] = calculate_coord_by_mid_point('bottom_left', 
        'top_right', 'top_left')
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
    source_points = np.float32([top_left_point, top_right_point, bottom_right_point, bottom_left_point])
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
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    final_labels = [labels[idx] for idx in pick]
    final_boxes = boxes[pick].astype("int")
    return final_boxes, final_labels

def sort_text(detection_boxes, detection_labels):
    detection_labels = np.array(detection_labels)
    id_boxes = detection_boxes[detection_labels == 1]
    name_boxes = detection_boxes[detection_labels == 2]
    birth_boxes = detection_boxes[detection_labels == 3]
    sex_boxes = detection_boxes[detection_labels == 4]
    national_boxes = detection_boxes[detection_labels == 5]
    home_boxes = detection_boxes[detection_labels == 6]
    add_boxes = detection_boxes[detection_labels == 7]
    exp_boxes = detection_boxes[detection_labels == 8]
    # arrange boxes
    id_boxes = sort_each_category(id_boxes)
    name_boxes = sort_each_category(name_boxes)
    birth_boxes = sort_each_category(birth_boxes)
    sex_boxes = sort_each_category(sex_boxes)
    national_boxes = sort_each_category(national_boxes)
    home_boxes = sort_each_category(home_boxes)
    add_boxes = sort_each_category(add_boxes)
    exp_boxes = sort_each_category(exp_boxes)
    return id_boxes, name_boxes, birth_boxes, sex_boxes, national_boxes, home_boxes, add_boxes, exp_boxes

def get_y1(x):
    return x[0]

def get_x1(x):
    return x[1]

def sort_each_category(category_text_boxes):
    min_y1 = min(category_text_boxes, key=get_y1)[0]
    mask = np.where(category_text_boxes[:, 0] < min_y1 + 10, True, False)
    line1_text_boxes = category_text_boxes[mask]
    line2_text_boxes = category_text_boxes[np.invert(mask)]
    line1_text_boxes = sorted(line1_text_boxes, key=get_x1)
    line2_text_boxes = sorted(line2_text_boxes, key=get_x1)
    if len(line2_text_boxes) != 0:
        merged_text_boxes = [*line1_text_boxes, *line2_text_boxes]
    else:
        merged_text_boxes = line1_text_boxes
    return merged_text_boxes
