# Grad-CAM Implementation

This repository contains code for implementing Grad-CAM (Gradient-weighted Class Activation Mapping) in PyTorch. Grad-CAM is a technique for visualizing the regions of an image that are important for predicting a particular class. It works by leveraging the gradients of a target class with respect to the ViT Norm layer.

## Requirements
- Python 3.x
- PyTorch
- Torchvision
- Timm
- NumPy
- OpenCV
- Scikit-image

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Mikael17125/ViT-GradCAM.git
   cd ViT-GradCAM
   ```

## Environment
1. Create the conda environment
   ```bash
   conda create --name vit-grad-cam python=3.10
   ```

2. Activate the environment
   ```bash
   conda activate vit-grad-cam
   ```

3. Install the PyTorch
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. Install additional dependency
   ```bash
   pip install -r requirements.txt   
   ```

## Usage
1. Ensure that your image is named `both.png` and placed in the root directory of the repository.

2. Run the `main.py` script:
   ```bash
   python main.py
   ```

3. The resulting heatmap with Grad-CAM overlay will be saved as `result.jpg` in the same directory.

## Files
- `main.py`: Python script to generate Grad-CAM visualization for a given image using a pre-trained Vision Transformer model (`vit_base_patch16_224`).
- `gradcam.py`: Python module containing the `GradCam` class, which implements the Grad-CAM algorithm.
- `both.png`: Sample input image (replace with your own image).
- `result.jpg`: Output Grad-CAM visualization.

## Input and Output
| Input Image | Output Image |
|-------------|--------------|
| ![Input Image](both.png) | ![Output Image](result.jpg) |

## References
- Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017.