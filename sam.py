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

# Chat function for HugChat
def chat_function(message):
    response = chatbot.chat(message).wait_until_done()
    return response

# Generate image and get quality score based on model selection
def generate_image_with_quality(prompt, selected_model):
    image, generated_caption, similarity_score = stable_diffusion.generate_image(prompt)
    
    # Get the quality score based on selected model
    if selected_model == "Prompt Similarity":
        score = stable_diffusion.get_prompt_similarity_score(prompt, generated_caption)
    elif selected_model == "CNNIQA":
        score = stable_diffusion.get_cnniqa_score(image)
    else:
        score = 0  # Placeholder for TRCNN or any other future model
    
    return image, score * 100  # Return percentage score

# Upload image and calculate accuracy
def handle_upload(image, selected_model):
    if selected_model == "CNNIQA":
        accuracy = stable_diffusion.get_cnniqa_score(image.name)
    else:
        accuracy = 0  # Handle other models if needed
    return Image.open(image.name), accuracy * 100

# Main frontend using Gradio Blocks
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
    .dropdown {
        border: 2px solid #4F46E5;
        border-radius: 8px;
        padding: 10px;
        margin: 10px;
        width: 100%;
        max-width: 500px;
        background-color: white;
        color: #4F46E5;
        font-family: Arial, sans-serif;
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
            
            # Adding the Dropdown for model selection
            with gr.Row():
                quality_model_selection = gr.Dropdown(
                    label="Select Quality Model",
                    choices=["CNNIQA", "Prompt Similarity", "TRCNN"],
                    value="Prompt Similarity",
                    interactive=True,
                    elem_id="dropdown"
                )
                quality_text = gr.Textbox(label="Image Quality", interactive=True, elem_id="textbox")

    # Update the chatbox based on the user input
    def update_chat(chatbox, message):
        response = chat_function(message)
        return chatbox + [(message, response)]

    # Handle image generation based on the selected quality model
    def handle_image_generation(prompt, selected_model):
        image, score = generate_image_with_quality(prompt, selected_model)
        return image, score

    # Handle image upload and quality score based on model selection
    def handle_uploaded_image(image, selected_model):
        image, accuracy = handle_upload(image, selected_model)
        return image, accuracy

    chat_button.click(update_chat, [chatbox, chat_input], chatbox)
    generate_button.click(handle_image_generation, [prompt_input, quality_model_selection], [image_display, quality_text])
    upload_button.upload(handle_uploaded_image, [upload_button, quality_model_selection], [image_display, quality_text])

demo.launch()
