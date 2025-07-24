import numpy as np
import torch
import cv2
import config
import matplotlib.pyplot as plt
import os


def plot_gradcam_on_eeg(eeg_signal: torch.Tensor, gradcam_map: np.ndarray, epoch: int,
                        save_path: str, sample_idx: int = 0):
    """
    Plots the EEG signal with GradCAM overlay.

    Args:
        eeg_signal: Tensor (C, T)
        gradcam_map: Array (T,)
    """
    C, T = eeg_signal.shape
    time = np.arange(T)

    plt.figure(figsize=(14, 2 * C))
    for i in range(C):
        plt.subplot(C, 1, i + 1)
        plt.plot(time, eeg_signal[i].cpu().numpy(), label=f'Channel {i+1}')
        plt.fill_between(time, 0, gradcam_map, color='red', alpha=0.3, label='GradCAM')
        plt.legend()
        plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'epoch_{epoch}_gradcam_sample_{sample_idx}.png'))
    plt.close()


def data_augmentation(data: torch.Tensor, label: torch.Tensor):
    """
    Perform segment-level mix augmentation.

    Args:
        data: Tensor (N, C, T)
        label: Tensor (N,)

    Returns:
        aug_data: Tensor (M, 1, C, T)
        aug_label: Tensor (M,)
    """
    device = data.device
    aug_data = []
    aug_label = []

    if data.dim() == 4 and data.shape[1] == 1:
        data = data.squeeze(1)

    N, C, T = data.shape
    seg_size = T // config.num_segs
    aug_data_size = config.batch_size // 4

    for cls in range(4):
        cls_idx = torch.where(label == cls)[0]
        if len(cls_idx) < 2:
            continue
        cls_data = data[cls_idx]
        temp_aug_data = torch.zeros((aug_data_size, C, T), device=device)

        for i in range(aug_data_size):
            rand_idx = torch.randint(0, cls_data.shape[0], (config.num_segs,), device=device)
            for j in range(config.num_segs):
                temp_aug_data[i, :, j * seg_size:(j + 1) * seg_size] = \
                    cls_data[rand_idx[j], :, j * seg_size:(j + 1) * seg_size]

        aug_data.append(temp_aug_data)
        aug_label.extend([cls] * aug_data_size)

    if not aug_data:
        return data, label

    aug_data = torch.cat(aug_data, dim=0).unsqueeze(1)
    aug_label = torch.tensor(aug_label, device=device)

    shuffle_idx = torch.randperm(len(aug_data), device=device)
    aug_data = aug_data[shuffle_idx]
    aug_label = aug_label[shuffle_idx]

    return aug_data, aug_label


class ActivationsAndGradients:
    """
    Helper to extract activations and gradients from intermediate layers.
    """

    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.activations = []
        self.gradients = []

        self.handles = []
        for layer in target_layers:
            self.handles.append(layer.register_forward_hook(self.save_activation))
            if hasattr(layer, 'register_full_backward_hook'):
                self.handles.append(layer.register_full_backward_hook(self.save_gradient))
            else:
                self.handles.append(layer.register_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform:
            grad = self.reshape_transform(grad)
        self.gradients.insert(0, grad.cpu().detach())

    def __call__(self, x):
        self.activations = []
        self.gradients = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    """
    Vanilla GradCAM for 1D EEG signals.

    Produces CAM of shape (N, T)
    """

    def __init__(self, model, target_layers, reshape_transform=None, use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=2, keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = sum(output[i, target_category[i]] for i in range(len(target_category)))
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        return np.sum(weights * activations, axis=1)

    @staticmethod
    def get_target_width_height(input_tensor):
        return input_tensor.size(-1), input_tensor.size(-2)

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().numpy() for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for acts, grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(acts, grads)
            cam[cam < 0] = 0
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = (img - np.min(img)) / (1e-7 + np.max(img - np.min(img)))
            if target_size:
                img = cv2.resize(img, target_size)
            result.append(img)
        return np.float32(result)

    def __call__(self, input_tensor, target_category):
        if self.cuda:
            input_tensor = input_tensor.cuda()
        output = self.activations_and_grads(input_tensor)
        targets = [target_category] * input_tensor.size(0)
        loss = self.get_loss(output, targets)
        print('GradCAM loss:', loss.item())
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return np.mean(np.concatenate(cam_per_layer, axis=1), axis=1)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.activations_and_grads.release()
        if exc_value and isinstance(exc_value, IndexError):
            print(f"[GradCAM Exception] {exc_type}: {exc_value}")
            return True


