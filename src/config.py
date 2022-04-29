corner_detection = {
    'path_to_model': '/home/whoisltd/detect/training/exported-models/ctc_chip/corner/saved_model',
    'path_to_labels': '/home/whoisltd/detect/data/ctc_chip/corner/label_map.pbtxt',
    'nms_ths': 0.2,
    'score_ths': 0.3
}

text_detection = {
    'path_to_model': '/home/whoisltd/detect/training/exported-models/ctc_chip/text/saved_model',
    'path_to_labels': '/home/whoisltd/detect/data/ctc_chip/text/label_map.pbtxt',
    'nms_ths': 0.2,
    'score_ths': 0.2
}

text_recognition = {
    'weight': '/home/whoisltd/detect/src/vietocr/transformerocr.pth'
}