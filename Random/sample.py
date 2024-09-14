import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class CNNIQAnet(nn.Module):
    def __init__(self, ker_size: int = 7, n_kers: int = 50, n1_nodes: int = 800, n2_nodes: int = 800):
        super(CNNIQAnet, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kers, ker_size)
        self.fc1 = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        h = self.conv1(x)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)
        h = h.squeeze(3).squeeze(2)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        q = torch.sigmoid(self.fc3(h)) * 100 # Scale to a smaller range
        return q

def load_model(model_path: str) -> CNNIQAnet:
    model = CNNIQAnet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path: str) -> torch.Tensor:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    img = cv2.resize(img, (64, 64))
    img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return img

def evaluate_image(model: CNNIQAnet, image_path: str) -> float:
    img = preprocess_image(image_path)
    with torch.no_grad():
        quality_score = model(img)
    return quality_score.item()

def main():
    model_path = 'Random/CNNIQA-LIVE (1)'
    image_path = 'Random/blur.jpeg'

    model = load_model(model_path)
    score = evaluate_image(model, image_path)
    print(f"The quality score for the image is: {score}")

if __name__ == "__main__":
    main()