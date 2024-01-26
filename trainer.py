import glob
import logging

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

import wandb
from data.dataset import BusiDataset, BratsDataset


def count_class(dataset, class_names):
    return [(dataset["image_path"].str.count(class_name) >= 1).sum() for class_name in class_names]


def print_class_pct(subset_name, class_names, subset):
    LOGGER = logging.getLogger(__name__)
    base_str = f"{subset_name}"
    classes_counts = count_class(subset, class_names)
    for i, class_name in enumerate(class_names):
        base_str += f" {class_name}: {classes_counts[i] / len(subset)} |"

    LOGGER.info(base_str)


def get_data(dataset_path):
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
    def __init__(self, model, optimizer, criterion, dataset_name,
                 dataset_path, batch_size, device, num_epochs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.log_interval = 8
        self.dataset_name = dataset_name

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
        self.best_dice = 0.0
        self.best_epoch = 0

    def get_dataset(self, dataset_name, dataset_path):
        LOGGER = logging.getLogger(__name__)
        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        mask_transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        if dataset_name == "BUSI":
            train_df, test_df, val_df = get_data(dataset_path)
            train_dataset = BusiDataset(train_df, self.device, image_processor, mask_transforms)
            test_dataset = BusiDataset(test_df, self.device, image_processor, mask_transforms)
            val_dataset = BusiDataset(val_df, self.device, image_processor, mask_transforms)
            LOGGER.info(f"Length Train Dataset: {len(train_dataset)}")
            LOGGER.info(f"Length Test Dataset: {len(test_dataset)}")
            LOGGER.info(f"Length Val Dataset: {len(val_dataset)}")
            return train_dataset, test_dataset, val_dataset
        elif dataset_name == "BRATS":
            train_df, test_df, val_df = get_data(dataset_path)
            train_dataset = BratsDataset(train_df, self.device, image_processor, mask_transforms)
            test_dataset = BratsDataset(test_df, self.device, image_processor, mask_transforms)
            return train_dataset, test_dataset
        else:
            raise ValueError(f"Unrecognized Dataset named {dataset_name}")

    def save_best_model(self, epoch, dice):
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
        # Training loop
        LOGGER = logging.getLogger(__name__)
        LOGGER.info(f"Start training on {self.device}")
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_dice = 0.0

            self.model.train()
            # LOGGER.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Learning Rate: {self.optimizer.param_groups[0]['lr']}")
            for i, (images, masks) in enumerate(pbar := tqdm(self.train_loader)):
                # free_mem, total_mem = torch.cuda.mem_get_info(0)
                # LOGGER.info(
                #     f"BATCH {i + 1}/{n_batch} | USED MEMORY {((total_mem - free_mem) / total_mem) * 100}%")
                images, masks = images.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice = 1 + self.criterion.dice(outputs, masks)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_dice += dice.item()
                pbar.set_postfix({"Loss": torch.round(loss, decimals=4).item()})

            avg_train_loss = running_loss / len(self.train_loader)
            avg_train_dice = running_dice / len(self.train_loader)
            LOGGER.info(
                f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}')
            wandb.log({"train loss": avg_train_loss, "train dice": avg_train_dice}, step=epoch)

            # Save metrics
            self.train_losses.append(avg_train_loss)
            self.train_dices.append(avg_train_dice)
            if epoch % 10 == 0:
                self.validate(epoch)
                self.val_epochs.append(epoch+1)
        with open('train_loss.npy', 'wb') as f:
            np.save(f, self.train_losses, allow_pickle=True)
        with open('val_loss.npy', 'wb') as f:
            np.save(f, self.val_losses, allow_pickle=True)

    @torch.no_grad()
    def test(self):
        running_dice = 0.0
        LOGGER = logging.getLogger(__name__)
        LOGGER.info(f"Start testing on {self.device}")
        self.model.eval()
        for i, (images, masks) in enumerate(pbar := tqdm(self.test_loader)):
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)
            dice = 1 + self.criterion.dice(outputs, masks)
            running_dice += dice
            pbar.set_postfix({"Dice coefficient": torch.round(dice, decimals=4).item()})

        avg_dice = running_dice / len(self.test_loader)
        LOGGER.info(f"Test Dice: {avg_dice}")
        wandb.log({"test dice": avg_dice})

    @torch.no_grad()
    def validate(self, epoch):
        LOGGER = logging.getLogger(__name__)
        running_loss = 0.0
        running_dice = 0.0

        self.model.eval()

        for i, (images, masks) in enumerate(pbar := tqdm(self.val_loader)):
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)

            loss = self.criterion(outputs, masks)
            dice = 1 + self.criterion.dice(outputs, masks)

            running_loss += loss.item()
            running_dice += dice.item()
            pbar.set_postfix({"Val Loss": torch.round(loss, decimals=4).item()})

        avg_val_loss = running_loss / len(self.train_loader)
        avg_val_dice = running_dice / len(self.train_loader)
        LOGGER.info(f'Epoch [{epoch}/{self.num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Dice {avg_val_dice:.4f}')

        # Save metrics
        self.val_losses.append(avg_val_loss)
        self.val_dices.append(avg_val_dice)
        wandb.log({"val loss": avg_val_loss, "val dice": avg_val_dice}, step=epoch)

        # Save best vitMaemodel
        self.save_best_model(epoch, avg_val_dice)

    def get_metrics(self):
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
