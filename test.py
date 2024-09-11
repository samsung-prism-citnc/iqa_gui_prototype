import torch
from torchvision import transforms
from PIL import Image

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval() 
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])
    image_tensor = preprocess(image).unsqueeze(0)  
    return image_tensor

# Function to predict image quality
def predict_image_quality(model, image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():  
        output = model(image_tensor)  
    return output.item() 

# Example usage
model_path = 'CNNIQA-LIVE'  
image_path = 'network_code.png'  

model = load_model(model_path)
quality_score = predict_image_quality(model, image_path)
print(f'Predicted Quality Score: {quality_score}')