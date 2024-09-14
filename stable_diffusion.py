import requests
from utils import base64_to_pil
import json
import os
import tempfile
from PIL import Image

STABLE_DIFFUSION_ENDPOINT = 'http://127.0.0.1:5000/generate_image'
CNNIQA_ENDPOINT = 'http://127.0.0.1:5000/evaluate_cnniqa'
PROMPT_SIMILARITY_ENDPOINT = 'http://127.0.0.1:5000/evaluate_prompt_similarity'

class StableDiffusion:
    def __init__(self):
        self.endpoint = STABLE_DIFFUSION_ENDPOINT
        self.cnniqa_endpoint = CNNIQA_ENDPOINT
        self.prompt_similarity_endpoint = PROMPT_SIMILARITY_ENDPOINT

    def generate_image(self, prompt):
        try:
            response = requests.get(self.endpoint, params={'prompt': prompt})
            response.raise_for_status()
            result = response.json()

            image_base64 = result.get('image_base64')
            generated_caption = result.get('generated_caption')
            similarity_score = result.get('similarity_score', 0)
            image = base64_to_pil(image_base64)

            return image, generated_caption, similarity_score
        except requests.RequestException as e:
            print(f"Error generating image: {e}")
            return None, None, 0

    def get_cnniqa_score(self, image: Image.Image):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg') as temp_file:
                temp_file_path = temp_file.name
                image.save(temp_file_path, format='JPEG')

            with open(temp_file_path, 'rb') as img_file:
                files = {'image': img_file}
                response = requests.post(self.cnniqa_endpoint, files=files)
                response.raise_for_status()
                result = response.json()

            os.remove(temp_file_path)

            score = result.get('cnniqa_quality_score', 0)
            return score
        except requests.RequestException as e:
            print(f"Error getting CNNIQA score: {e}")
            return 0

    def get_prompt_similarity_score(self, prompt, generated_caption):
        try:
            payload = {
                'prompt': prompt,
                'generated_caption': generated_caption
            }
            response = requests.post(self.prompt_similarity_endpoint, json=payload)
            response.raise_for_status()
            result = response.json()

            return result.get('similarity_score', 0)
        except requests.RequestException as e:
            print(f"Error getting prompt similarity score: {e}")
            return 0    