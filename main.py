import logging
import os
import random

import numpy as np
import torch
from monai.losses import DiceCELoss

from models.mae_unetr import MaeUnetr
from trainer import Trainer
from utilis.utils import plot_metrics, plot_subplots

# from utilis.utils import diceLoss, bce_dice_loss, dice_coef_loss

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


# def show_image(image, title=''):
#     # image is [H, W, 3]
#     assert image.shape[2] == 3
#     plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
#     plt.title(title, fontsize=16)
#     plt.axis('off')
#     return
#
#
# def visualize(pixel_values, vitMaemodel):
#     # global imagenet_mean
#     # global imagenet_std
#     # imagenet_mean = np.array(image_processor.image_mean)
#     # imagenet_std = np.array(image_processor.image_std)
#
#     # forward pass
#     outputs = vitMaemodel(pixel_values)
#     y = vitMaemodel.unpatchify(outputs.logits)
#     # questa cosa mistica cambia solo l'ordine della size
#     y = torch.einsum('nchw->nhwc', y).detach().cpu().squeeze(0)
#     show_image(y, "reconstruction")
#     # plt.imshow(y.squeeze(0).numpy())
#     # plt.show()
#
#     # # visualize the mask
#     # mask = outputs.mask.detach()
#     # mask = mask.unsqueeze(-1).repeat(1, 1, vitMaemodel.config.patch_size ** 2 * 3)  # (N, H*W, p*p*3)
#     # mask = vitMaemodel.unpatchify(mask)  # 1 is removing, 0 is keeping
#     # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
#     #
#     # x = torch.einsum('nchw->nhwc', pixel_values)
#     #
#     # # masked image
#     # im_masked = x * (1 - mask)
#     #
#     # # MAE reconstruction pasted with visible patches
#     # im_paste = x * (1 - mask) + y * mask
#     #
#     # # make the plt figure larger
#     # plt.rcParams['figure.figsize'] = [24, 24]
#     #
#     # plt.subplot(1, 4, 1)
#     # show_image(x[0], "original")
#     #
#     # plt.subplot(1, 4, 2)
#     # show_image(im_masked[0], "masked")
#     #
#     # plt.subplot(1, 4, 3)
#     # show_image(y[0], "reconstruction")
#     #
#     # plt.subplot(1, 4, 4)
#     # show_image(im_paste[0], "reconstruction + visible")
#     #
#     plt.show()


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

    # 1- MODEL
    model = MaeUnetr()
    model.to(device)

    # 2- DATA LOADING AND TRAINER
    # With trainer you define also datasets and data loaders
    busi_trainer = Trainer(
        model=model,
        batch_size=64,
        device=device,
        num_epochs=100,
        optimizer=torch.optim.Adam(lr=1e-4, weight_decay=1e-5, params=model.get_parameters()),
        # optimizer=torch.optim.Adam(lr=1e-4, params=model.get_parameters()),
        criterion=DiceCELoss(to_onehot_y=True,
                             softmax=True,
                             squared_pred=False),
        dataset_name="BUSI",
        dataset_path="/media/data/lbonanni/Dataset_BUSI_with_GT"
    )

    # 3- TRAINING
    busi_trainer.train()
    metrics = busi_trainer.get_metrics()
    plot_metrics(metrics)

    # 4- TESTING
    busi_trainer.test()

    for i in [2, 3, 10, 20]:
        image = busi_trainer.test_dataset[i][0]
        mask = busi_trainer.test_dataset[i][1]
        im = image.to(device)
        pred = model(im.unsqueeze(0))
        pred = pred.squeeze()

        plot_subplots(im, mask, pred)

    # inputs['pixel_values']

    # visualize(inputs['pixel_values'], vitMaemodel)

    # plt.imshow(a.argmax(dim=1).squeeze(0).detach().numpy(), cmap='gray')
    # plt.show()
