import logging
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from transformers import ViTMAEModel, AutoImageProcessor
from unetr_decoder import UNETR_decoder


SEED = 5
LOGGER = logging.getLogger(__name__)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def seed_everything(seed: int):
    LOGGER.info(f"Seed: {seed}")
    if seed is None:
        seed = random.randint(1, 10000)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    # opt = get_opt()
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    seed_everything(SEED)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    inputs = image_processor(images=image, return_tensors="pt")
    encoder.embeddings.config.mask_ratio = 0
    encoder.config.output_hidden_states = True
    outputs = encoder(pixel_values=inputs['pixel_values'])
    decoder = UNETR_decoder(
        in_channels=3,
        out_channels=1,
        img_size=(224, 224),  # TODO replace
        patch_size=16,
        feature_size=16,
        hidden_size=768,
        spatial_dims=2
    )
    hidden_states = tuple(hs[:, 1:, :] for hs in outputs.hidden_states)
    a = decoder(
        x=inputs['pixel_values'],
        x_in=inputs['pixel_values'],
        hidden_states_out=hidden_states
    )
    plt.matshow(a.squeeze(0, 1).detach().numpy())
    plt.show()
    print()
