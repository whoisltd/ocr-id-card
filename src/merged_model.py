import numpy as np

from PIL import Image
from src.detector.detector import Detector
# from src.vietocr.text_recognition import TextRecognition
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from src.detector.utils.image_utils import align_image, sort_text
from src.config import text_detection


class CompletedModel(object):
    def __init__(self):
    #     self.corner_detection_model = Detector(path_to_model=corner_detection['path_to_model'],
    #                                            path_to_labels=corner_detection['path_to_labels'],
    #                                            nms_threshold=corner_detection['nms_ths'], 
    #                                            score_threshold=corner_detection['score_ths'])
        self.text_detection_model = Detector(path_to_model=text_detection['path_to_model'],
                                             path_to_labels=text_detection['path_to_labels'],
                                             nms_threshold=text_detection['nms_ths'], 
                                             score_threshold=text_detection['score_ths'])
        # self.text_recognition_model = TextRecognition()

        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = '/home/whoisltd/detect/src/vietocr/config_text_recognition/transformerocr.pth'
        config['cnn']['pretrained']=False
        config['device'] = 'cpu'
        config['predictor']['beamsearch']=False 
        self.detector = Predictor(config)
        # init boxes
        self.num_boxes = None
        self.name_boxes = None
        self.birth_boxes = None
        self.hometown_boxes = None
        self.addr_boxes = None

    # def detect_corner(self, image):
    #     detection_boxes, detection_classes, category_index = self.corner_detection_model.predict(image)

    #     coordinate_dict = dict()
    #     height, width, _ = image.shape

    #     for i in range(len(detection_classes)):
    #         label = str(category_index[detection_classes[i]]['name'])
    #         real_ymin = int(max(1, detection_boxes[i][0]))
    #         real_xmin = int(max(1, detection_boxes[i][1]))
    #         real_ymax = int(min(height, detection_boxes[i][2]))
    #         real_xmax = int(min(width, detection_boxes[i][3]))
    #         coordinate_dict[label] = (real_xmin, real_ymin, real_xmax, real_ymax)

    #     # align image
    #     cropped_img = align_image(image, coordinate_dict)

    #     return cropped_img
    
    def detect_text(self, image):
        # detect text boxes
        detection_boxes, detection_classes, _ = self.text_detection_model.predict(image)
        # print(detection_boxes)
        # print(detection_classes)
        # sort text boxes according to coordinate
        self.num_boxes, self.name_boxes, self.birth_boxes, self.hometown_boxes, self.addr_boxes = sort_text(detection_boxes, detection_classes)

    def recognize(self, image):
        field_dict = dict()

        # crop boxes according to coordinate
        def crop_and_recog(boxes):
            crop = []
            if len(boxes) == 1:
                ymin, xmin, ymax, xmax = boxes[0].astype(np.int32)
                crop.append(image[ymin:ymax, xmin:xmax])
            else:
                for box in boxes:
                    ymin, xmin, ymax, xmax = box.astype(np.int32)
                    crop.append(image[ymin:ymax, xmin:xmax])

            return crop

        list_ans = list(crop_and_recog(self.num_boxes))
        list_ans.extend(crop_and_recog(self.name_boxes))
        list_ans.extend(crop_and_recog(self.birth_boxes))
        list_ans.extend(crop_and_recog(self.hometown_boxes))
        list_ans.extend(crop_and_recog(self.addr_boxes))

        list_ans = [Image.fromarray(i) for i in list_ans]
        result = self.detector.predict_batch(list_ans)
            # result = self.detector.predict(np.array(list_ans))
        field_dict['id'] = result[0]
        field_dict['name'] = ' '.join(result[1:len(self.name_boxes) + 1])
        field_dict['birth'] = result[len(self.name_boxes) + 1]
        field_dict['hometown'] = ' '.join(result[len(self.name_boxes) + 2: -len(self.hometown_boxes)])
        field_dict['addr'] = ' '.join(result[-len(self.hometown_boxes):])

        return field_dict

    def predict(self, image):
        # cropped_image = self.detect_corner(image)
        cropped_image = image
        self.detect_text(image)
        # print(self.)
        return self.recognize(image)