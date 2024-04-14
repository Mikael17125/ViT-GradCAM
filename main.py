import timm
import torch
from skimage import io
from torchvision.models import vgg19, resnet18
from torchsummary import summary
from gradcam import GradCam
import numpy as np
import cv2

timm.list_models('vit_*')

def prepare_input(image):
    image = image.copy()

    means = np.array([0.5, 0.5, 0.5])
    stds = np.array([0.5, 0.5, 0.5])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = image[np.newaxis, ...]

    return torch.tensor(image, requires_grad=True)

def gen_cam(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = (1 - 0.5) * heatmap + 0.5 * image
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    img = io.imread("both.png")
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = prepare_input(img)

    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    target_layer = model.blocks[-1].norm1
    
    grad_cam = GradCam(model, target_layer)
    mask = grad_cam(inputs)
    result = gen_cam(img, mask)

    cv2.imwrite('result.jpg', result)
