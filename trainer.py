import glob
import logging

import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

from data.dataset import BusiDataset, BratsDataset


# from utilis.utils import dice_coeff

def get_data(dataset_path):
    masks = glob.glob(f"{dataset_path}/*/*_mask.png")
    images = [mask_images.replace("_mask", "") for mask_images in masks]
    series = list(zip(images, masks))
    dataset = pd.DataFrame(series, columns=['image_path', 'mask_path'])
    train, test = train_test_split(dataset, test_size=0.25)
    return train, test


class Trainer:
    def __init__(self, model, optimizer, criterion, dataset_name,
                 dataset_path, batch_size, device, num_epochs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.log_interval = 8

        self.train_dataset, self.test_dataset = self.get_dataset(dataset_name, dataset_path)
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True)

        # Lists to store training and validation metrics
        self.train_losses = []
        self.test_losses = []
        self.train_dices = []
        self.test_dices = []

        # Best vitMaemodel and its metrics
        self.best_model = None
        self.best_dice = 0.0
        self.best_epoch = 0

    def get_dataset(self, dataset_name, dataset_path):
        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        mask_transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        if dataset_name == "BUSI":
            train_df, test_df = get_data(dataset_path)
            train_dataset = BusiDataset(train_df, self.device, image_processor, mask_transforms)
            test_dataset = BusiDataset(test_df, self.device, image_processor, mask_transforms)
            return train_dataset, test_dataset
        elif dataset_name == "BRATS":
            train_df, test_df = get_data(dataset_path)
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

            self.model.train()

            for i, (images, masks) in enumerate(pbar := tqdm(self.train_loader)):
                # free_mem, total_mem = torch.cuda.mem_get_info(0)
                # LOGGER.info(
                #     f"BATCH {i + 1}/{n_batch} | USED MEMORY {((total_mem - free_mem) / total_mem) * 100}%")
                batch_size = images.size(0)
                images, masks = images.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                # pbar.set_postfix(f"Loss {loss.item() / batch_size:.4f}")
                pbar.set_postfix({"Loss": torch.round(loss, decimals=4).item()})

            avg_train_loss = running_loss / len(self.train_loader)
            LOGGER.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}')

            # Save metrics
            self.train_losses.append(avg_train_loss)

            # Save best vitMaemodel
            self.save_best_model(epoch + 1, avg_train_loss)

    def test(self):
        running_loss = 0.0
        LOGGER = logging.getLogger(__name__)
        LOGGER.info(f"Start testing on {self.device}")
        self.model.eval()
        with torch.no_grad():
            for i, (images, masks) in enumerate(pbar := tqdm(self.test_loader)):
                images, masks = images.to(self.device), masks.to(self.device)
                batch_size = images.size(0)

                outputs = self.model(images)
                dice = self.criterion.dice(outputs, masks)
                running_loss += dice
                pbar.set_postfix({"Dice coefficient": torch.round(dice, decimals=4).item()})
                # LOGGER.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}')
                # print(f'Batch {i}: Dice coefficient {dice/batch_size:.4f}')

        # print(f'Total dice coefficient {test_dice/i:.4f}')

    def get_metrics(self):
        return {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_dices': self.train_dices,
            'test_dices': self.test_dices,
            'best_model': self.best_model,
            'best_dice': self.best_dice,
            'best_epoch': self.best_epoch
        }
