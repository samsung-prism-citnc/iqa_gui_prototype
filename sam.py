import gradio as gr
from PIL import Image
from stable_diffusion import StableDiffusion
from hugchat import hugchat
from hugchat.login import Login
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

# HugChat credentials and login setup
EMAIL = "sahilgowda204@gmail.com"
PASSWD = "Sahilgowda2004"
cookie_path_dir = "./cookies/"

sign = Login(EMAIL, PASSWD)
cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)
chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

stable_diffusion = StableDiffusion()

def chat_function(message):
    response = chatbot.chat(message).wait_until_done()
    return response

def generate_image(prompt):
    image, generated_caption, similarity_score = stable_diffusion.generate_image(prompt)
    return image, similarity_score * 100  # Return the similarity score as a percentage

# Load the pre-trained CNN models for IQA
# Replace 'path_to_model' with the actual path to your pre-trained models
model_trcnn = tf.keras.models.load_model('path_to_trcnn_model')
model_hiqa = tf.keras.models.load_model('path_to_hiqa_model')
model_cnniqa = tf.keras.models.load_model('path_to_cnniqa_model')

def calculate_quality_score(image_path, model):
    """
    Calculate the quality score of an image using a pre-trained CNN model.

    Args:
        image_path (str): The path to the image file.
        model (tf.keras.Model): The pre-trained CNN model.

    Returns:
        float: The quality score of the image.
    """
    # Load the image and preprocess it
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict the quality score using the CNN model
    quality_score = model.predict(x)[0][0]

    # Return the quality score
    return quality_score

def handle_upload(image, model_name):
    # Select the appropriate model based on the model name
    if model_name == "TRCNN":
        model = model_trcnn
    elif model_name == "HIQA":
        model = model_hiqa
    elif model_name == "CNNIQA":
        model = model_cnniqa
    else:
        raise ValueError("Invalid model name")

    # Calculate the quality score using the selected model
    quality_score = calculate_quality_score(image.name, model)
    return Image.open(image.name), quality_score  # Return the image and quality score

with gr.Blocks(css="""
    .container {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        justify-content: center;
        padding: 20px;
    }
    .header {
        border: 2px solid #000;
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        font-family: Arial, sans-serif;
        margin-bottom: 20px;
    }
    .box {
        border: 2px solid #4F46E5;
        border-radius: 8px;
        padding: 10px;
        margin: 10px;
        width: 100%;
        max-width: 500px;
        height: 430px;
    }
    .button {
        background-color: #3B3A9A;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
        margin-top: 10px;
    }
    .button:hover {
        background-color: #2C287A;
    }
    .textbox {
        border: 2px solid #4F46E5;
        border-radius: 8px;
        padding: 10px;
        margin: 10px;
        width: 100%;
        max-width: 500px;
    }
""") as demo:
    with gr.Row(elem_id="container"):
        with gr.Column(scale=1):
            gr.HTML("<div class='header'><h2>Chat Box</h2></div>")
            chatbox = gr.Chatbot(height=750, elem_id="box")
            chat_input = gr.Textbox(show_label=False, placeholder="Type a message...", elem_id="textbox")
            chat_button = gr.Button("Send", elem_id="button")

        with gr.Column(scale=2):
            gr.HTML("<div class='header'><h2>Image Display</h2></div>")
            prompt_input = gr.Textbox(placeholder="Enter the prompt for image generation...", label="Prompt", elem_id="textbox")
            image_display = gr.Image(label="Image Display", type="pil", height=430, elem_id="box")
            with gr.Row():
                generate_button = gr.Button("Generate Image", elem_id="button")
                upload_button = gr.UploadButton("Upload Image", elem_id="button")
            model_selector = gr.Dropdown(choices=["TRCNN", "HIQA", "CNNIQA"], label="Select Model", elem_id="textbox")
            quality_text_trcnn = gr.Textbox(label="TRCNN Quality", interactive=False, elem_id="textbox")
            quality_text_hiqa = gr.Textbox(label="HIQA Quality", interactive=False, elem_id="textbox")
            quality_text_cnniqa = gr.Textbox(label="CNNIQA Quality", interactive=False, elem_id="textbox")

    def update_chat(chatbox, message):
        response = chat_function(message)
        return chatbox + [(message, response)]

    def handle_image_generation(prompt):
        image, similarity_score = generate_image(prompt)
        return image, similarity_score

    def handle_model_selection(image, model_name):
        image, quality_score = handle_upload(image, model_name)
        if model_name == "TRCNN":
            return image, quality_score, "", ""
        elif model_name == "HIQA":
            return image, "", quality_score, ""
        elif model_name == "CNNIQA":
            return image, "", "", quality_score

    chat_button.click(update_chat, [chatbox, chat_input], chatbox)
    generate_button.click(handle_image_generation, prompt_input, [image_display, quality_text_trcnn])
    upload_button.upload(handle_model_selection, [upload_button, model_selector], [image_display, quality_text_trcnn, quality_text_hiqa, quality_text_cnniqa])

demo.launch()