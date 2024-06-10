import requests
from utils import base64_to_pil

STABLE_DIFFUSION_ENDPOINT = 'http://anssug-ip-34-122-150-57.tunnelmole.net/'

class StableDiffusion:
    def __init__(self):
        self.endpoint = STABLE_DIFFUSION_ENDPOINT

    def generate_image(self, prompt):
        response = requests.get(self.endpoint, params={'prompt': prompt})
        image_base64 = response.content.decode('utf-8')
        image = base64_to_pil(image_base64)
        return image