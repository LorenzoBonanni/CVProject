import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_transform=None, mask_transform=None):
        """
        Dataset class for the BUSI (Breast Ultrasound Images) dataset.

        This dataset class loads images and masks from a provided DataFrame.
        It allows custom image and mask transformations to be applied during loading.

        :param dataframe: The DataFrame containing image and mask paths.
        :type dataframe: pd.DataFrame
        :param image_transform: The transformation to apply to images.
        :type image_transform: callable, optional
        :param mask_transform: The transformation to apply to masks.
        :type mask_transform: callable, optional
        """
        self.data = dataframe
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        """
        Get the length of the dataset.

        :return: The number of samples in the dataset.
        :rtype: int
        """
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset.

        :param idx: The index of the sample to retrieve.
        :type idx: int
        :return: The image and its corresponding mask.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        image_path = self.data.iloc[idx]['image_path']
        mask_path = self.data.iloc[idx]['mask_path']

        image = Image.open(image_path)
        if image.mode == "L":
            rgbImage = Image.new("RGB", image.size)
            rgbImage.paste(image)
            image = rgbImage

        if not image.mode == "RGB":
            image = image.convert("RGB")

        mask = Image.open(mask_path)
        if mask.mode == "RGB":
            mask = mask.convert("L")

        if self.image_transform:
            image = self.image_transform(images=image, return_tensors="pt")['pixel_values']
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image.squeeze(0), mask
