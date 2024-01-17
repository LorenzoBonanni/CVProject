import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from utilis.utils import DiceLoss
from transformers import ViTMAEForPreTraining

from trainer import Trainer
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


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def visualize(pixel_values, model):
    # global imagenet_mean
    # global imagenet_std
    # imagenet_mean = np.array(image_processor.image_mean)
    # imagenet_std = np.array(image_processor.image_std)

    # forward pass
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    # questa cosa mistica cambia solo l'ordine della size
    y = torch.einsum('nchw->nhwc', y).detach().cpu().squeeze(0)
    show_image(y, "reconstruction")
    # plt.imshow(y.squeeze(0).numpy())
    # plt.show()

    # # visualize the mask
    # mask = outputs.mask.detach()
    # mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size ** 2 * 3)  # (N, H*W, p*p*3)
    # mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    #
    # x = torch.einsum('nchw->nhwc', pixel_values)
    #
    # # masked image
    # im_masked = x * (1 - mask)
    #
    # # MAE reconstruction pasted with visible patches
    # im_paste = x * (1 - mask) + y * mask
    #
    # # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [24, 24]
    #
    # plt.subplot(1, 4, 1)
    # show_image(x[0], "original")
    #
    # plt.subplot(1, 4, 2)
    # show_image(im_masked[0], "masked")
    #
    # plt.subplot(1, 4, 3)
    # show_image(y[0], "reconstruction")
    #
    # plt.subplot(1, 4, 4)
    # show_image(im_paste[0], "reconstruction + visible")
    #
    plt.show()


if __name__ == '__main__':
    # opt = get_opt()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        handlers=[
            # logging.FileHandler(f'{base_path}/{opt["numberepochs"]}/INFO.log', 'w'),
            logging.StreamHandler()
        ]
    )
    seed_everything(SEED)
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    for param in model.parameters():
        param.requires_grad = False

    vitmae_encoder = model.vit
    vitmae_encoder.embeddings.config.mask_ratio = 0
    vitmae_encoder.config.output_hidden_states = True
    decoder = UNETR_decoder(
        in_channels=3,
        out_channels=2,
        img_size=(224, 224),  # TODO replace
        patch_size=16,
        feature_size=16,
        hidden_size=768,
        spatial_dims=2
    )

    trainer = Trainer(
        encoder=model,
        decoder=decoder,
        optimizer=torch.optim.Adam(lr=1e-4, weight_decay=1e-5, params=decoder.parameters()),
        criterion=DiceLoss(),
        device=device
    )


    # inputs['pixel_values']

    # visualize(inputs['pixel_values'], model)

    # plt.imshow(a.argmax(dim=1).squeeze(0).detach().numpy(), cmap='gray')
    # plt.show()
