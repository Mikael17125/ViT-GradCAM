import cv2
import numpy as np


class GradCam:
    def __init__(self, model, target):
        self.model = model.eval()
        self.feature = None
        self.gradient = None
        self.handlers = []
        self.target = target
        self._get_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = self.reshape_transform(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = self.reshape_transform(output_grad)

        def _store_grad(grad):  
            self.gradient = self.reshape_transform(grad)

        output_grad.register_hook(_store_grad)
    
    def _get_hook(self):
        self.target.register_forward_hook(self._get_features_hook)
        self.target.register_forward_hook(self._get_grads_hook)

    def reshape_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                          height, 
                                          width, 
                                          tensor.size(2))

        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def __call__(self, inputs):
            
        self.model.zero_grad()
        output = self.model(inputs) 
        
        index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))
        feature = self.feature[0].cpu().data.numpy()

        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0) 
        cam = np.maximum(cam, 0) 

        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (224, 224))
        return cam