import gradio as gr
from PIL import Image
from stable_diffusion import StableDiffusion
from hugchat import hugchat
from hugchat.login import Login

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

def check_image_quality(image, prompt):
    similarity_score = get_similarity_score(image, prompt)
    return f"Similarity Score: {similarity_score * 100:.2f}%"

def get_similarity_score(image, prompt):
    # Placeholder function for getting similarity score
    # Replace this with actual model inference code
    return 0.85  # Example similarity score

with gr.Blocks(css="""
    .container {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        justify-content: center;
        padding: 20px;
    }
    .header {
        border: 2px solid #4F46E5;
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
        background-color: #4F46E5;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
        margin-top: 10px;
    }
    .button:hover {
        background-color: #3B3A9A;
    }
""") as demo:
    with gr.Row(elem_id="container"):
        with gr.Column(scale=1):
            gr.HTML("<div class='header'><h2>Chat Box</h2></div>")
            chatbox = gr.Chatbot(height=750)
            chat_input = gr.Textbox(show_label=False, placeholder="Type a message...")
            chat_button = gr.Button("Send", elem_id="button")

        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.HTML("<div class='header'><h2>Generated Image</h2></div>")
                    prompt_input = gr.Textbox(placeholder="Enter the prompt for image generation...", label="Prompt")
                    generated_image = gr.Image(label="Generated Image", type="pil", height=430, elem_id="box")
                    generate_button = gr.Button("Generate Image", elem_id="button")
                    check_generated_quality_button = gr.Button("Check Quality", elem_id="button")

                with gr.Column():
                    gr.HTML("<div class='header'><h2>Uploaded Image</h2></div>")
                    upload_button = gr.File(label="Upload Image", type="filepath")
                    upload_image = gr.Image(label="Uploaded Image", type="pil", height=430, elem_id="box")
                    check_upload_quality_button = gr.Button("Check Quality", elem_id="button")

            with gr.Row():
                with gr.Column():
                    generated_quality_text = gr.Textbox(label="Generated Image Quality", interactive=False)

                with gr.Column():
                    upload_quality_text = gr.Textbox(label="Uploaded Image Quality", interactive=False)

    def update_chat(chatbox, message):
        response = chat_function(message)
        return chatbox + [(message, response)]

    def handle_image_generation(prompt):
        image, similarity_score = generate_image(prompt)
        return image, similarity_score

    def handle_upload(image):
        return Image.open(image.name)

    def perform_quality_check(image, prompt):
        result = check_image_quality(image, prompt)
        return result

    chat_button.click(update_chat, [chatbox, chat_input], chatbox)
    generate_button.click(handle_image_generation, prompt_input, [generated_image, generated_quality_text])
    upload_button.upload(handle_upload, upload_button, upload_image)
    check_generated_quality_button.click(perform_quality_check, [generated_image, prompt_input], generated_quality_text)
    check_upload_quality_button.click(perform_quality_check, [upload_image, prompt_input], upload_quality_text)

demo.launch()
