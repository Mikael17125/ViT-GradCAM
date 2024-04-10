import cv2
import numpy as np
import timm
from skimage import io

from gradcam import GradCam

def prepare_input(image):
    """
    Preprocesses the input image to make it compatible with the model.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        torch.Tensor: The preprocessed image as a PyTorch tensor.
    """
    means = np.array([0.5, 0.5, 0.5])
    stds = np.array([0.5, 0.5, 0.5])
    image = ((image.astype(np.float32) / 255) - means) / stds
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image[np.newaxis, ...], requires_grad=True)

def gen_cam(image, mask):
    """
    Generates the Class Activation Map (CAM) using the input image and mask.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        mask (numpy.ndarray): The Grad-CAM mask as a NumPy array.

    Returns:
        numpy.ndarray: The Class Activation Map as a NumPy array.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    cam = (1 - 0.5) * heatmap + 0.5 * image
    return np.uint8(255 * cam / np.max(cam))

if __name__ == '__main__':
    # Load the input image
    img = io.imread("both.png")
    img = np.float32(cv2.resize(img, (224, 224))) / 255

    # Prepare the input for the model
    inputs = prepare_input(img)

    # Load the Vision Transformer model
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    target_layer = model.blocks[-1].norm1
    
    # Initialize the Grad-CAM
    grad_cam = GradCam(model, target_layer)

    # Compute the Grad-CAM mask
    mask = grad_cam(inputs)

    # Generate the Class Activation Map
    result = gen_cam(img, mask)

    # Save the resulting image
    cv2.imwrite('result.jpg', result)
