import requests
from utils import base64_to_pil
import json

STABLE_DIFFUSION_ENDPOINT = 'http://127.0.0.1:5000/generate_image'

class StableDiffusion:
    def __init__(self):
        self.endpoint = STABLE_DIFFUSION_ENDPOINT

    def generate_image(self, prompt):
        response = requests.get(self.endpoint, params={'prompt': prompt})
        result = response.json()

        image_base64 = result['image_base64']
        generated_caption = result['generated_caption']
        similary_score = result['similarity_score']
        image = base64_to_pil(image_base64)
        
        return image, generated_caption, similary_score