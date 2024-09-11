import gradio as gr
from PIL import Image
from stable_diffusion import StableDiffusion
from hugchat import hugchat
from hugchat.login import Login
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np

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

def calculate_accuracy_vgg16(image_path):
    """
    Calculate the accuracy of an image using a pre-trained VGG16 model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        float: The accuracy of the image as a percentage.
    """
    # Load the pre-trained VGG16 model
    model = VGG16(weights='imagenet')

    # Load the image and preprocess it
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions on the image
    preds = model.predict(x)

    # Get the predicted label
    predicted_label = np.argmax(preds[0])

    # For demonstration purposes, we'll assume the accuracy is based on the confidence of the predicted label
    accuracy = preds[0][predicted_label]

    # Return the accuracy as a percentage
    return accuracy * 100

def handle_upload(image):
    # Calculate the accuracy using VGG16
    accuracy = calculate_accuracy_vgg16(image.name)
    return Image.open(image.name), accuracy  # Return the image and accuracy as a percentage

with gr.Blocks(css="""
    .container {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        justify-content: center;
        padding: 20px;
    }
    .header {
        border:2px solid rgb(0 0 0 / 44%);
        background-color: rgb(168 198 255);
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
            quality_text = gr.Textbox(label="Image Quality", interactive=False, elem_id="textbox")

    def update_chat(chatbox, message):
        response = chat_function(message)
        return chatbox + [(message, response)]

    def handle_image_generation(prompt):
        image, similarity_score = generate_image(prompt)
        return image, similarity_score

    chat_button.click(update_chat, [chatbox, chat_input], chatbox)
    generate_button.click(handle_image_generation, prompt_input, [image_display, quality_text])
    upload_button.upload(handle_upload, upload_button, [image_display, quality_text])

demo.launch()