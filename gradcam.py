import cv2
import numpy as np

class GradCam:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation for visualizing
    important regions in an input image based on the activations of a specific target layer.

    Args:
        model (torch.nn.Module): The trained neural network model.
        target (torch.nn.Module): The target layer in the model for which Grad-CAM will be computed.

    Attributes:
        model (torch.nn.Module): The trained neural network model.
        feature (torch.Tensor): Feature maps extracted from the target layer.
        gradient (torch.Tensor): Gradients of the target layer's output with respect to model's output.
        target (torch.nn.Module): The target layer in the model for which Grad-CAM will be computed.
    """

    def __init__(self, model, target):
        self.model = model.eval()
        self.feature = None
        self.gradient = None
        self.target = target
        self._get_hook()

    def _get_features_hook(self, module, input, output):
        """
        Hook function to extract feature maps from the target layer.
        """
        self.feature = self.reshape_transform(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        Hook function to compute gradients of the target layer's output with respect to model's output.
        """
        self.gradient = self.reshape_transform(output_grad)

        def _store_grad(grad):  
            self.gradient = self.reshape_transform(grad)

        output_grad.register_hook(_store_grad)
    
    def _get_hook(self):
        """
        Register hooks for extracting feature maps and computing gradients of the target layer.
        """
        self.target.register_forward_hook(self._get_features_hook)
        self.target.register_forward_hook(self._get_grads_hook)

    def reshape_transform(self, tensor, height=14, width=14):
        """
        Reshape and transpose the tensor to facilitate Grad-CAM computation.
        """
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                          height, 
                                          width, 
                                          tensor.size(2))

        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def __call__(self, inputs):
        """
        Compute Grad-CAM for the given input image.

        Args:
            inputs (torch.Tensor): The input image tensor.

        Returns:
            numpy.ndarray: The computed Grad-CAM heatmap.
        """
        self.model.zero_grad()
        output = self.model(inputs) 
        
        index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))
        feature = self.feature[0].cpu().data.numpy()

        # Theorem for Grad-CAM computation:
        # Grad-CAM = Σ(∂L/∂A_k * A_k)
        # Where:
        # ∂L/∂A_k : Gradient of the target class activation with respect to the feature maps of the target layer.
        # A_k : Feature maps of the target layer.
        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0) 
        cam = np.maximum(cam, 0) 

        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (224, 224))
        return cam
