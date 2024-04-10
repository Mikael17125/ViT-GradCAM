import cv2
import numpy as np
import timm
from skimage import io

from gradcam import GradCam  # Importing GradCam class from gradcam.py

def prepare_input(image):
    """
    Preprocesses the input image for compatibility with the model.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.

    Returns:
        torch.Tensor: Preprocessed image as a PyTorch tensor.
    """
    image = image.copy()

    means = np.array([0.5, 0.5, 0.5])
    stds = np.array([0.5, 0.5, 0.5])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = image[np.newaxis, ...]

    return torch.tensor(image, requires_grad=True)

def gen_cam(image, mask):
    """
    Generates the class activation map (CAM) using the input image and mask.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        mask (numpy.ndarray): Grad-CAM mask as a NumPy array.

    Returns:
        numpy.ndarray: Class activation map as a NumPy array.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = (1 - 0.5) * heatmap + 0.5 * image
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    # Load the input image
    img = io.imread("both.png")
    img = np.float32(cv2.resize(img, (224, 224))) / 255

    # Prepare input for the model
    inputs = prepare_input(img)

    # Load Vision Transformer model
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    target_layer = model.blocks[-1].norm1
    
    # Initialize Grad-CAM
    grad_cam = GradCam(model, target_layer)

    # Compute Grad-CAM mask
    mask = grad_cam(inputs)

    # Generate class activation map
    result = gen_cam(img, mask)

    # Save the resulting image
    cv2.imwrite('result.jpg', result)
