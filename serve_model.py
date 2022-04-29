from io import BytesIO
from json.tool import main
import numpy as np
from PIL import Image
from src.merged_model import CompletedModel
model = None


def load_model():
    model = CompletedModel()
    return model


def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()

    img = np.asarray(image)
    result = model.predict(img)

    return result


def read_image_file(file) -> Image.Image:
    image = Image.open(file)

    return image

image = read_image_file('/home/whoisltd/Desktop/as12.jpg')
print(predict(image))