import logging
import os
import random

import numpy as np
import torch
from transformers import ViTMAEModel, AutoImageProcessor
from PIL import Image
import requests

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
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    inputs = image_processor(images=image, return_tensors="pt")
    model.embeddings.config.mask_ratio = 0
    outputs = model(pixel_values=inputs['pixel_values'])
    last_hidden_states = outputs.last_hidden_state
    # print()
    # train(opt)
    # test(opt)
