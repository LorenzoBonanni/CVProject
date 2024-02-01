import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from models.mae_unetr import MaeUnetr
from models.unet import Unet


def get_model(model_name: str, opt):
    """
    Retrieve a specific model instance based on the provided model name and options.

    :param model_name: Name of the model to retrieve. Supported values include "MAE_UNETR" and "UNET".
    :type model_name: str
    :param opt: Command line arguments passed to the program.
    :type opt: argparse.Namespace
    :return: An instance of the specified model.
    :rtype: nn.Module
    :raises ValueError: If the provided model_name is unrecognized.
    """
    if model_name == "MAE_UNETR":
        return MaeUnetr(opt.train_mae)
    elif model_name == "UNET":
        return Unet()
    else:
        raise ValueError(f"Unrecognized Model named {model_name}")


# def plot_train_label(image, mask):
#     f, axarr = plt.subplots(1, 3, figsize=(5, 5))
#
#     axarr[0].imshow(np.squeeze(image), cmap='gray', origin='lower')
#     axarr[0].set_ylabel('Axial View', fontsize=14)
#     axarr[0].set_xticks([])
#     axarr[0].set_yticks([])
#     axarr[0].set_title('CT', fontsize=14)
#
#     axarr[1].imshow(np.squeeze(mask), cmap='jet', origin='lower')
#     axarr[1].axis('off')
#     axarr[1].set_title('Mask', fontsize=14)
#
#     axarr[2].imshow(np.squeeze(image), cmap='gray', alpha=1, origin='lower')
#     axarr[2].imshow(np.squeeze(mask), cmap='jet', alpha=0.5, origin='lower')
#     axarr[2].axis('off')
#     axarr[2].set_title('Overlay', fontsize=14)
#
#     plt.tight_layout()
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.show()


def plot_metrics(metrics):
    """
    Plot training and validation metrics over epochs.

    Given a dictionary containing various metrics related to the training process of a model,
    this function generates plots for training and validation losses as well as training and
    validation dice loss over epochs.

    :param metrics: A dictionary containing the following keys:
                    - 'train_losses': A list of training losses for each epoch.
                    - 'val_losses': A list of validation losses for each epoch.
                    - 'train_dices': A list of training dice loss for each epoch.
                    - 'val_dices': A list of validation dice loss for each epoch.
                    - 'val_epochs': A list specifying the epochs at which validation
                                    losses and dice coefficients were calculated.
    :type metrics: dict
    :return: The matplotlib figure object containing the generated plots.
    :rtype: matplotlib.figure.Figure
    """
    num_epochs = len(metrics['train_losses'])
    epochs = np.arange(1, num_epochs + 1)
    epochs_val = metrics['val_epochs']

    # Convert tensors to NumPy arrays
    train_losses_np = metrics['train_losses']
    val_losses_np = metrics['val_losses']
    train_dices_np = metrics['train_dices']
    val_dices_np = metrics['val_dices']

    # Plot Losses
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_np, label='Train Loss')
    plt.plot(epochs_val, val_losses_np, label='Val Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Dice Coefficients
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_dices_np, label='Train Dice')
    plt.plot(epochs_val, val_dices_np, label='Val Dice')
    plt.title('Training Dice Coefficients')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.tight_layout()
    plt.show()
    return fig


def to_numpy(tensor):
    """
    Convert a PyTorch tensor to a NumPy array.

    This function moves the input tensor to the CPU (if not already) and detaches it from
    the computational graph before converting it to a NumPy array.

    :param tensor: The input PyTorch tensor to be converted.
    :type tensor: torch.Tensor
    :return: The NumPy array representation of the input tensor.
    :rtype: np.ndarray
    """
    return tensor.cpu().detach().numpy()


def plot_subplots(image, mask, predicted, imagenet_mean, imagenet_std):
    """
    Plot the image, mask, and predicted segmentation mask as subplots.

    This function takes the input image, ground truth mask, and predicted segmentation mask
    (along with imagenet_mean and imagenet_std for normalization) and plots them as subplots.

    :param image: The input image tensor.
    :type image: torch.Tensor
    :param mask: The ground truth mask tensor.
    :type mask: torch.Tensor
    :param predicted: The predicted segmentation mask tensor.
    :type predicted: torch.Tensor
    :param imagenet_mean: The mean values used for image normalization.
    :type imagenet_mean: np.ndarray
    :param imagenet_std: The standard deviation values used for image normalization.
    :type imagenet_std: np.ndarray
    :return: The matplotlib figure object containing the plotted subplots.
    :rtype: matplotlib.figure.Figure
    """
    # Convert tensors to NumPy arrays
    image_np, mask_np, predicted_np = map(to_numpy, (image, mask, predicted))

    # Threshold the predicted values
    predicted_np_thresholded = np.expand_dims(np.argmax(predicted_np, axis=0), 0)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Adjust figsize as needed

    # Plot Image, Mask, Predicted, and Thresholded Predicted
    titles = ['Image', 'Mask', 'Predicted']
    for ax, data, title in zip(axes, [image_np, mask_np, predicted_np_thresholded], titles):
        if title == 'Image':
            data = np.clip((data * np.expand_dims(imagenet_std, (1, 2)) + np.expand_dims(imagenet_mean, (1, 2))) * 255,0, 255).astype(int)
        data = np.transpose(data, (1, 2, 0))
        ax.imshow(data, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.show()
    return fig


def boolean_string(s):
    """
    Convert a string representation of boolean value to a boolean.

    This function takes a string representation of a boolean value ('True' or 'False')
    and converts it to a boolean type.

    :param s: The string representation of a boolean value.
    :type s: str
    :return: The boolean value represented by the input string.
    :rtype: bool
    :raises ValueError: If the input string is not 'True' or 'False'.
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of epochs (integer value)", default=200)
    parser.add_argument("--lr", type=float, help="Learning rate (float value)", default=1e-4)
    parser.add_argument("--decay_factor", type=float, help="Decay Factor (float value)", default=0.8)
    parser.add_argument("--scheduler", type=boolean_string, help="Set scheduler flag to True", default=False)
    parser.add_argument("--train_mae", type=boolean_string, help="Set scheduler flag to True", default=False)
    parser.add_argument("--mae_epochs", type=int, help="Number of epochs (integer value)", default=0)
    parser.add_argument("--mae_lr", type=float, help="Learning rate (float value)", default=1e-4)
    parser.add_argument("--device", type=int, help="number of device", default=0)
    parser.add_argument("--batch_size", type=int, help="size of the batch", default=64)
    parser.add_argument("--model_name", type=str, help="name of the model to train", default="MAE_UNETR")
    parser.add_argument("--dataset", type=str, help="name of the dataset", default="BUSI")
    parser.add_argument("--dataset_path", type=str, help="path to the dataset",
                        default="/media/data/lbonanni/Dataset_BUSI_with_GT")
    parser.add_argument("--num_exp", type=int, help="number of experiments to perform", default=5)
    args = parser.parse_args()
    args.epochs -= args.mae_epochs
    return args
