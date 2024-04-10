import timm
import torch
from skimage import io
from torchvision.models import vgg19, resnet18, resnet50, resnet152
from torchsummary import summary
from gradcam import GradCam
import numpy as np
import cv2

timm.list_models('vit_*')

def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = image[np.newaxis, ...]

    return torch.tensor(image, requires_grad=True)

def norm_image(image):
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]

    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)

if __name__ == '__main__':
    img = io.imread("both.png")
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = prepare_input(img)

    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    print(model)
    
    grad_cam = GradCam(model)
    mask = grad_cam(inputs)
    a, b = gen_cam(img, mask)

    cv2.imwrite('result.jpg', a)
    cv2.imwrite('mask.jpg', b)
