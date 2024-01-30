import glob
import logging

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from monai.metrics import DiceMetric, HausdorffDistanceMetric

import wandb
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

from data.dataset import BusiDataset


def count_class(dataset: pd.DataFrame, class_names: list[str]):
    """
    Count occurrences of each class in the dataset.

    This function counts the occurrences of each class specified in `class_names`
    within the dataset. It checks the 'image_path' column of the dataset for each
    class occurrence.

    :param dataset: The dataset containing the images.
    :type dataset: pandas.DataFrame
    :param class_names: A list of class names to count occurrences for.
    :type class_names: list[str]
    :return: A list containing the count of occurrences for each class in `class_names`.
    :rtype: list[int]
    """
    return [(dataset["image_path"].str.count(class_name) >= 1).sum() for class_name in class_names]


def print_class_pct(subset_name: str, class_names: list[str], subset: pd.DataFrame):
    """
    Print the percentage of each class in the given subset.

    This function calculates and prints the percentage of each class in the given subset
    of data.

    :param subset_name: Name of the subset being analyzed (e.g., 'TRAIN', 'TEST', 'VALIDATION').
    :type subset_name: str
    :param class_names: List of class names to analyze.
    :type class_names: list[str]
    :param subset: Subset of data to analyze.
    :type subset: pandas.DataFrame
    """
    LOGGER = logging.getLogger(__name__)
    base_str = f"{subset_name}"
    classes_counts = count_class(subset, class_names)
    for i, class_name in enumerate(class_names):
        base_str += f" {class_name}: {classes_counts[i] / len(subset)} |"

    LOGGER.info(base_str)


def get_busi_data(dataset_path: str):
    """
    Load and preprocess data for the BUSI dataset.

    This function reads images and masks from the specified dataset path, splits them into
    training, testing, and validation subsets, and prints the class distribution percentages
    for each subset.

    :param dataset_path: Path to the directory containing the BUSI dataset.
    :type dataset_path: str
    :return: Tuple containing the training, testing, and validation subsets of the dataset.
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    masks = glob.glob(f"{dataset_path}/*/*_mask.png")
    images = [mask_images.replace("_mask", "") for mask_images in masks]
    series = list(zip(images, masks))
    dataset = pd.DataFrame(series, columns=['image_path', 'mask_path'])
    train, test = train_test_split(dataset, test_size=0.15, shuffle=True)
    train, val = train_test_split(train, test_size=0.214285714, shuffle=True)  # 0.214285714×0.7 = 0.15
    classes = ['benign', 'normal', 'malign']
    print_class_pct("TRAIN", classes, train)
    print_class_pct("TEST", classes, test)
    print_class_pct("VAL", classes, val)

    return train, test, val


class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, dataset_name: str,
                 dataset_path: str, batch_size: int, device: torch.device, num_epochs: int,
                 scheduler: bool, decay_factor: float, start_lr: float):
        """
        A class to manage training and evaluation of a neural network model.

        This class facilitates the training and evaluation of a neural network model
        using the specified optimizer, loss criterion, and dataset. Additionally, it tracks
        and stores training and validation metrics such as losses and dice coefficients.

        :param model: The neural network model to be trained.
        :type model: torch.nn.Module
        :param optimizer: The optimizer used for training the model.
        :type optimizer: torch.optim.Optimizer
        :param criterion: The loss criterion used for optimization.
        :type criterion: torch.nn.Module
        :param dataset_name: The name of the dataset being used.
        :type dataset_name: str
        :param dataset_path: The path to the dataset.
        :type dataset_path: str
        :param batch_size: The batch size used during training.
        :type batch_size: int
        :param device: The device (CPU or GPU) on which the model will be trained.
        :type device: torch.device
        :param num_epochs: The number of training epochs.
        :type num_epochs: int
        :param scheduler: Whether to use a learning rate scheduler during training.
        :type scheduler: bool
        :param decay_factor: The decay factor used by the learning rate scheduler.
        :type decay_factor: float
        :param start_lr: The learning rate used by the scheduler.
        :type start_lr: float
        """
        self.imagenet_std = None
        self.imagenet_mean = None
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.log_interval = 8
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        self.train_dataset, self.test_dataset, self.val_dataset = self.get_dataset(dataset_name, dataset_path)
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True)

        # Lists to store training and validation metrics
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.train_dices = []
        self.test_dices = []
        self.val_dices = []
        self.val_epochs = []

        # Best vitMaemodel and its metrics
        self.best_model = None
        self.best_dice = 0
        self.best_epoch = 0
        self.use_scheduler = scheduler
        self.decay_factor = decay_factor
        self.start_lr = start_lr

    def get_dataset(self, dataset_name: str, dataset_path: str):
        """
        Load and preprocess the dataset.

        This method loads and preprocesses the dataset specified by `dataset_name` from the
        directory `dataset_path`. It applies appropriate transformations to the images and
        masks based on the dataset requirements.

        :param dataset_name: The name of the dataset to load.
        :type dataset_name: str
        :param dataset_path: The path to the directory containing the dataset.
        :type dataset_path: str
        :return: Tuple containing the training, testing, and validation datasets.
        :rtype: tuple[Dataset, Dataset, Dataset]
        :raises ValueError: If the dataset name is unrecognized.
        """
        LOGGER = logging.getLogger(__name__)
        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        self.imagenet_mean = np.array(image_processor.image_mean)
        self.imagenet_std = np.array(image_processor.image_std)
        mask_transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        if dataset_name == "BUSI":
            train_df, test_df, val_df = get_busi_data(dataset_path)
            train_dataset = BusiDataset(train_df, image_processor, mask_transforms)
            test_dataset = BusiDataset(test_df, image_processor, mask_transforms)
            val_dataset = BusiDataset(val_df, image_processor, mask_transforms)
            LOGGER.info(f"Length Train Dataset: {len(train_dataset)}")
            LOGGER.info(f"Length Test Dataset: {len(test_dataset)}")
            LOGGER.info(f"Length Val Dataset: {len(val_dataset)}")
            return train_dataset, test_dataset, val_dataset
        elif dataset_name == "BRATS":
            # train_df, test_df, val_df = get_data(dataset_path)
            # train_dataset = BratsDataset(train_df, self.device, image_processor, mask_transforms)
            # test_dataset = BratsDataset(test_df, self.device, image_processor, mask_transforms)
            # return train_dataset, test_dataset
            pass
        else:
            raise ValueError(f"Unrecognized Dataset named {dataset_name}")

    def save_best_model(self, epoch: int, dice: float):
        """
        Save the best model based on the dice coefficient.

        This method saves the model parameters if the provided `dice` loss
        is better than the current best dice loss. It also updates the
        best dice loss and epoch.

        :param epoch: The current epoch number.
        :type epoch: int
        :param dice: The dice coefficient obtained during the current epoch.
        :type dice: float
        """
        if dice > self.best_dice:
            LOGGER = logging.getLogger(__name__)
            LOGGER.info(f"Saved New Model at Epoch {epoch}")
            self.best_dice = dice
            self.best_epoch = epoch
            self.best_model = self.model.state_dict()

            log_directory = '/media/data/lbonanni/Dataset_BUSI_with_GT/pretrained'
            # os.makedirs(log_directory, exist_ok=True)

            # Specify the file path for saving the vitMaemodel
            filename = f'{log_directory}/best_model_epoch{epoch}_dice{dice:.4f}.pth'
            torch.save(self.best_model, filename)

    def train(self):
        """
        Train the neural network model.

        This method performs the training loop for the neural network model over
        multiple epochs. It updates the model parameters using the specified optimizer
        and loss criterion. Additionally, it logs training metrics such as loss and
        dice loss, and periodically validates the model on the validation dataset.

        """
        # Training loop
        LOGGER = logging.getLogger(__name__)
        LOGGER.info(f"Start training on {self.device}")
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_dice = 0.0
            lr = self.optimizer.param_groups[0]['lr']
            if self.use_scheduler:
                self.optimizer.param_groups[0]['lr'] = self.start_lr * (self.decay_factor ** (epoch // 20))
                # self.optimizer.param_groups[0]['lr'] = self.start_lr * (self.decay_factor ** (epoch // 20))
            wandb.log({"learning rate": lr}, step=epoch+1)

            self.model.train()
            for i, (images, masks) in enumerate(pbar := tqdm(self.train_loader)):
                images, masks = images.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice = 1 - self.criterion.dice(outputs, masks)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_dice += dice.item()
                pbar.set_postfix({"Loss": torch.round(loss, decimals=4).item()})

            avg_train_loss = running_loss / len(self.train_loader)
            avg_train_dice = running_dice / len(self.train_loader)
            LOGGER.info(
                f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}')
            wandb.log({"train loss": avg_train_loss, "train dice": avg_train_dice}, step=epoch+1)

            # Save metrics
            self.train_losses.append(avg_train_loss)
            self.train_dices.append(avg_train_dice)
            if (epoch + 1) % 10 == 0:
                self.validate(epoch+1)
                self.val_epochs.append(epoch + 1)

    @torch.no_grad()
    def test(self):
        """
        Evaluate the neural network model on the test dataset.

        This method evaluates the trained neural network model on the test dataset.
        It computes the dice loss for each batch of images and masks, and then
        calculates the average dice loss for the entire test dataset.
        """
        running_dice = 0.0
        LOGGER = logging.getLogger(__name__)
        LOGGER.info(f"Start testing on {self.device}")
        self.model.load_state_dict(self.best_model)
        self.model.eval()
        for i, (images, masks) in enumerate(pbar := tqdm(self.test_loader)):
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)
            dice = 1 - self.criterion.dice(outputs, masks)
            running_dice += dice
            pbar.set_postfix({"Dice coefficient": torch.round(dice, decimals=4).item()})

        avg_dice = running_dice / len(self.test_loader)
        LOGGER.info(f"Test Dice: {avg_dice}")
        # LOGGER.info(f"Test Hausdorff: {avg_hausdorff}")
        wandb.log({"test dice": avg_dice})
        # wandb.log({"test hausdorff": avg_hausdorff})

    @torch.no_grad()
    def validate(self, epoch: int):
        """
        Perform validation on the validation dataset.

        This method evaluates the trained neural network model on the validation dataset
        to assess its performance. It computes the loss and dice loss for each batch
        of images and masks, and then calculates the average loss and dice loss for
        the entire validation dataset.

        :param epoch: The current epoch number.
        :type epoch: int
        """
        LOGGER = logging.getLogger(__name__)
        running_loss = 0.0
        running_dice = 0.0

        self.model.eval()

        for i, (images, masks) in enumerate(pbar := tqdm(self.val_loader)):
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)

            loss = self.criterion(outputs, masks)
            dice = 1 - self.criterion.dice(outputs, masks)

            running_loss += loss.item()
            running_dice += dice.item()
            pbar.set_postfix({"Val Loss": torch.round(loss, decimals=4).item()})

        avg_val_loss = running_loss / len(self.val_loader)
        avg_val_dice = running_dice / len(self.val_loader)
        LOGGER.info(f'Epoch [{epoch}/{self.num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Dice {avg_val_dice:.4f}')

        # Save metrics
        self.val_losses.append(avg_val_loss)
        self.val_dices.append(avg_val_dice)
        wandb.log({"val loss": avg_val_loss, "val dice": avg_val_dice}, step=epoch)

        # Save best vitMaemodel
        self.save_best_model(epoch, avg_val_dice)

    def get_metrics(self):
        """
        Retrieve training and evaluation metrics.

        This method returns a dictionary containing various metrics tracked during the
        training and evaluation of the neural network model. The metrics include training
        and validation losses, training and validation dice loss, information
        about the best model (if saved), and related epoch information.

        :return: Dictionary containing the tracked metrics.
        :rtype: dict
        """
        return {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_dices': self.train_dices,
            'test_dices': self.test_dices,
            'val_losses': self.val_losses,
            'val_dices': self.val_dices,
            'best_model': self.best_model,
            'best_dice': self.best_dice,
            'best_epoch': self.best_epoch,
            'val_epochs': self.val_epochs
        }
