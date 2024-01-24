import matplotlib.pyplot as plt
import numpy as np


def plot_train_label(image, mask):
    f, axarr = plt.subplots(1, 3, figsize=(5, 5))

    axarr[0].imshow(np.squeeze(image), cmap='gray', origin='lower')
    axarr[0].set_ylabel('Axial View', fontsize=14)
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])
    axarr[0].set_title('CT', fontsize=14)

    axarr[1].imshow(np.squeeze(mask), cmap='jet', origin='lower')
    axarr[1].axis('off')
    axarr[1].set_title('Mask', fontsize=14)

    axarr[2].imshow(np.squeeze(image), cmap='gray', alpha=1, origin='lower')
    axarr[2].imshow(np.squeeze(mask), cmap='jet', alpha=0.5, origin='lower')
    axarr[2].axis('off')
    axarr[2].set_title('Overlay', fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_metrics(metrics):
    num_epochs = len(metrics['train_losses'])
    epochs = np.arange(1, num_epochs + 1)

    # Convert tensors to NumPy arrays
    train_losses_np = metrics['train_losses']
    # val_losses_np = metrics['val_losses']
    # train_dices_np = [to_numpy(dice) for dice in metrics['train_dices']]
    # val_dices_np = [to_numpy(dice) for dice in metrics['val_dices']]

    # Plot Losses
    plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_np, label='Train Loss')
    # plt.plot(epochs, val_losses_np, label='Val Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Dice Coefficients
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, train_dices_np, label='Train Dice')
    # plt.plot(epochs, val_dices_np, label='Val Dice')
    # plt.title('Training Dice Coefficients')
    # plt.xlabel('Epoch')
    # plt.ylabel('Dice Coefficient')
    # plt.legend()

    plt.tight_layout()
    plt.show()


def to_numpy(tensor):
    # Move tensor to CPU and convert to NumPy array
    return tensor.cpu().detach().numpy()


def plot_subplots(image, mask, predicted, threshold=0.5):
    # Convert tensors to NumPy arrays
    image_np, mask_np, predicted_np = map(to_numpy, (image, mask, predicted))

    # Threshold the predicted values
    predicted_np_thresholded = np.expand_dims(np.argmax(predicted_np, axis=0), 0)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Adjust figsize as needed

    # Plot Image, Mask, Predicted, and Thresholded Predicted
    titles = ['Image', 'Mask', 'Predicted']
    for ax, data, title in zip(axes, [image_np, mask_np, predicted_np_thresholded], titles):
        # data = torch.einsum('chw->hwc', data)
        data = np.transpose(data, (1, 2, 0))
        ax.imshow(data, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.show()
