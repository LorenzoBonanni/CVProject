from PIL import Image
from torch.utils.data import Dataset

class BusiDataset(Dataset):
    def __init__(self, dataframe, device, image_transform=None, mask_transform=None):
        self.data = dataframe
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_path']
        mask_path = self.data.iloc[idx]['mask_path']

        image = Image.open(image_path).convert('rgb')
        mask = Image.open(mask_path).convert('rgb')

        if self.image_transform:
            image = self.image_transform(images=image, return_tensors="pt")['pixel_values'].to(self.device)
        if self.mask_transform:
            mask = self.mask_transform(mask).to(self.device)

        return image, mask
