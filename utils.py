import base64
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from PIL import Image
from io import BytesIO

def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    image = Image.open(BytesIO(base64_decoded))
    return image


def pil_image_to_qpixmap(pil_image):
    image_array = np.array(pil_image)
    qimage = QImage(image_array.data, image_array.shape[1], image_array.shape[0], QImage.Format_RGB888).rgbSwapped()
    qpixmap = QPixmap.fromImage(qimage)
    return qpixmap