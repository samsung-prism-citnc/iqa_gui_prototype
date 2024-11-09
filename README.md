# Image Quality Assessment (IQA) with TRCNN Model

![WhatsApp Image 2024-10-26 at 12 58 10_cbb80fcf](https://github.com/user-attachments/assets/607bd045-dbd0-497e-b2e2-91e0cfde65f5)

## Overview

This project implements an **Image Quality Assessment (IQA)** model using a **TRCNN (Transformer-CNN)** architecture. The model combines the power of **Convolutional Neural Networks (CNNs)** for feature extraction and **Visual Transformers (VTs)** for ranking images based on their quality. The goal is to assess the perceptual quality of images across a wide range of conditions, such as compression artifacts, noise, and distortion.

### Key Components:
1. **CNN**: Used for extracting abstract, hierarchical features from the input images.
2. **Visual Transformer (VT)**: Handles the ranking task by attending to important image regions and providing a score based on quality.
3. **TRCNN Model**: The combination of CNN and VT for a robust image quality assessment system that can handle diverse image conditions.

## Features
- Efficient image quality ranking using deep learning.
- Integration of CNN and Transformer-based architectures for enhanced feature extraction and ranking.
- Pre-trained model support for faster results.
- Ability to evaluate images in various quality conditions.

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/image-quality-assessment.git
cd image-quality-assessment
```

### Requirements

Make sure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

### Dependencies:
- PyTorch (>=1.8)
- torchvision
- numpy
- matplotlib
- transformers
- OpenCV
- Pillow

## Model Architecture

### Convolutional Neural Network (CNN):
The CNN component is responsible for extracting essential features from the input images. The architecture leverages several layers of convolution, pooling, and activation functions to process the images and create a compact representation of the image content.

### Visual Transformer (VT):
The Transformer component is designed to handle the ranking of images. It uses self-attention mechanisms to focus on key areas of the image, leveraging global context and spatial relationships. The VT produces a quality score, ranking the image based on the features extracted by the CNN.

### TRCNN Model:
The TRCNN combines the CNN's feature extraction with the VT's ranking abilities to assess image quality. The CNN processes the image and provides its feature representation, which is then fed into the Transformer. The Transformer computes a ranking score that reflects the perceived quality of the image.

## Interface

![WhatsApp Image 2024-09-27 at 12 29 03_794eeff4](https://github.com/user-attachments/assets/726eb16c-8ebc-48b6-8a22-94da2e02e323)
![WhatsApp Image 2024-09-27 at 12 29 03_795b6675](https://github.com/user-attachments/assets/c85cb277-affd-4c16-b85c-2008265516f1)
![WhatsApp Image 2024-09-27 at 12 29 03_030897f9](https://github.com/user-attachments/assets/b0fa7926-f319-48a5-9a67-1936e4135844)

## Training

To train the model, use the following command:

```bash
python train.py --epochs <num_epochs> --batch-size <batch_size> --learning-rate <learning_rate>
```

### Arguments:
- `--epochs`: Number of training epochs.
- `--batch-size`: Size of each training batch.
- `--learning-rate`: Learning rate for the optimizer.

Make sure to provide your training dataset in the appropriate format (e.g., images and corresponding quality labels).

## Inference

For inference, you can use the pre-trained model or your trained model to assess the quality of an image:

```bash
python inference.py --image-path <image_path> --model-path <model_path>
```

### Arguments:
- `--image-path`: Path to the image you want to assess.
- `--model-path`: Path to the trained model file (if you are using a custom model).

The model will output a quality score, where a higher score indicates better image quality.

## Evaluation

To evaluate the performance of the model on a test dataset, run:

```bash
python evaluate.py --test-data <test_data_path> --model-path <model_path>
```

### Arguments:
- `--test-data`: Path to the test dataset (with images and corresponding ground truth labels).
- `--model-path`: Path to the trained model file.

The script will calculate performance metrics such as **Mean Squared Error (MSE)**, **Pearson Correlation Coefficient (PCC)**, and **Spearman Rank Correlation (SROCC)** for the image quality assessment.

## Example Usage

1. **Training the model:**

   ```bash
   python train.py --epochs 50 --batch-size 32 --learning-rate 0.0001
   ```

2. **Running inference on a single image:**

   ```bash
   python inference.py --image-path ./test_images/image1.jpg --model-path ./models/trcnn_model.pth
   ```

3. **Evaluating the model:**

   ```bash
   python evaluate.py --test-data ./test_data/ --model-path ./models/trcnn_model.pth
   ```

## Results

The model has shown promising results on several benchmark IQA datasets such as **LIVE** and **TID2013**, achieving competitive scores in terms of correlation with human perceptual judgments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The CNN and Transformer architectures are inspired by [insert relevant research papers].
- Thanks to the contributors of the PyTorch library for providing the deep learning framework.
