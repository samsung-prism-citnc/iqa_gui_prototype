import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# Define the CNNIQA model architecture
class CNNIQAnet(nn.Module):
    def __init__(self, ker_size: int = 7, n_kers: int = 50, n1_nodes: int = 800, n2_nodes: int = 800):
        super(CNNIQAnet, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kers, ker_size)
        self.fc1 = nn.Linear(2 * n_kers, n1_nodes)  # Adjusted for max-min pooling
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  # Reshape for Conv2d
        h = self.conv1(x)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))  # Max pooling
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))  # Min pooling
        h = torch.cat((h1, h2), 1)  # Concatenate max-min pooling results
        h = h.squeeze(3).squeeze(2)  # Flatten
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        q = self.fc3(h)
        return q

# Load the pre-trained model weights
def load_model(model_path: str) -> CNNIQAnet:
    model = CNNIQAnet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the input image
def preprocess_image(image_path: str) -> torch.Tensor:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = cv2.resize(img, (64, 64))  # Resize to 64x64
    img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)  # Normalize and add batch dimension
    return img

# Evaluate the image quality
def evaluate_image(model: CNNIQAnet, image_path: str) -> float:
    img = preprocess_image(image_path)
    with torch.no_grad():
        quality_score = model(img)  # Get the quality score
    return quality_score.item()  # Return the score as a float

# Main function to run the evaluation
def main():
    model_path = 'D:/iqa_gui_prototype/Random/CNNIQA-LIVE (1)'  # Update with the correct model path
    image_path = 'D:/iqa_gui_prototype/Random/image.webp'  # Update with the correct image path

    model = load_model(model_path)
    score = evaluate_image(model, image_path)
    print(f"The quality score for the image is: {score}")

if __name__ == "__main__":
    main()