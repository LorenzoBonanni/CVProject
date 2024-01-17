import glob

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor
from data.busi_dataset import BusiDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, encoder, decoder, optimizer, criterion, dataset_name, dataset_path, batch_size, device):
        self.encoder = encoder
        self.decoder = decoder
        self.train_dataset, self.test_dataset = self.get_dataset(dataset_name, dataset_path)
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=batch_size)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def get_dataset(self, dataset_name, dataset_path):
        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        mask_transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])
        if dataset_name == "BUSI":
            train_df, test_df = self.get_busi(dataset_path)
            train_dataset = BusiDataset(train_df, self.device, image_processor, mask_transforms)
            test_dataset = BusiDataset(test_df, self.device, image_processor, mask_transforms)
            return train_dataset, test_dataset
        else:
            raise ValueError(f"Unrecognized Dataset named {dataset_name}")

    def get_busi(self, dataset_path):
        masks = glob.glob(f"{dataset_path}/*/*_mask.png")
        images = [mask_images.replace("_mask", "") for mask_images in masks]
        series = list(zip(images, masks))
        dataset = pd.DataFrame(series, columns=['image_path', 'mask_path'])
        train, test = train_test_split(dataset, test_size=0.25)
        return train, test

    def forward(self, inputs):
        outputs = self.encoder(pixel_values=inputs)
        hidden_states = tuple(hs[:, 1:, :] for hs in outputs.hidden_states)
        a = self.decoder(
            x=self.encoder.unpatchify(outputs.logits),
            x_in=inputs,
            hidden_states_out=hidden_states
        )
