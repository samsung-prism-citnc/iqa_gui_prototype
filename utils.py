import base64
from PIL import Image
from io import BytesIO

def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    image = Image.open(BytesIO(base64_decoded))
    return image