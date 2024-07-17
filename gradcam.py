import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations

class GradCam:
    def __init__(self, model, target):
        self.model = model.eval()  # Set the model to evaluation mode
        self.feature = None  # To store the features from the target layer
        self.gradient = None  # To store the gradients from the target layer
        self.handlers = []  # List to keep track of hooks
        self.target = target  # Target layer for Grad-CAM
        self._get_hook()  # Register hooks to the target layer

    # Hook to get features from the forward pass
    def _get_features_hook(self, module, input, output):
        self.feature = self.reshape_transform(output)  # Store and reshape the output features

    # Hook to get gradients from the backward pass
    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = self.reshape_transform(output_grad)  # Store and reshape the output gradients

        def _store_grad(grad):
            self.gradient = self.reshape_transform(grad)  # Store gradients for later use

        output_grad.register_hook(_store_grad)  # Register hook to store gradients

    # Register forward hooks to the target layer
    def _get_hook(self):
        self.target.register_forward_hook(self._get_features_hook)
        self.target.register_forward_hook(self._get_grads_hook)

    # Function to reshape the tensor for visualization
    def reshape_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)  # Rearrange dimensions to (C, H, W)
        return result

    # Function to compute the Grad-CAM heatmap
    def __call__(self, inputs):
        self.model.zero_grad()  # Zero the gradients
        output = self.model(inputs)  # Forward pass

        # Get the index of the highest score in the output
        index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]  # Get the target score
        target.backward()  # Backward pass to compute gradients

        # Get the gradients and features
        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))  # Average the gradients
        feature = self.feature[0].cpu().data.numpy()

        # Compute the weighted sum of the features
        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0)  # Sum over the channels
        cam = np.maximum(cam, 0)  # Apply ReLU to remove negative values

        # Normalize the heatmap
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (224, 224))  # Resize to match the input image size
        return cam  # Return the Grad-CAM heatmap
