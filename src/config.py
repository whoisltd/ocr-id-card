text_detection = {
    'path_to_model': '/home/whoisltd/detect/src/detector/config_text_detection/model.tflite',
    'path_to_labels': '/home/whoisltd/detect/src/detector/config_text_detection/label_map.pbxt',
    'nms_ths': 0.2,
    'score_ths': 0.2

}

text_recognition = {
    'base_config': '/home/whoisltd/detect/src/vietocr/config_text_recognition/base.yml',
    'vgg_config': '/home/whoisltd/detect/src/vietocr/config_text_recognition/vgg-transformer.yml',
    'model_weight': '/home/whoisltd/detect/src/vietocr/config_text_recognition/transformerocr.pth'
}