import copy
import logging
import os
import random

import numpy as np
import torch
from monai.losses import DiceCELoss

import wandb
from models.mae_unetr import MaeUnetr
from trainer import Trainer
from utilis.utils import plot_metrics, plot_subplots, get_opt

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
    n_epoch, lr, scheduler, decay_factor = get_opt()
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
    trainer = Trainer(
        model=model,
        batch_size=64,
        device=device,
        num_epochs=n_epoch,
        optimizer=torch.optim.Adam(lr=lr, params=model.get_parameters()),
        criterion=DiceCELoss(to_onehot_y=True,
                             softmax=True,
                             squared_pred=False),
        dataset_name="BUSI",
        dataset_path="/media/data/lbonanni/Dataset_BUSI_with_GT",
        scheduler=scheduler,
        decay_factor=decay_factor,
        start_lr=copy.copy(lr)
    )
    run = wandb.init(
        project="UnetMae",
        config={
            "learning_rate": lr,
            "dataset": trainer.dataset_name,
            "epochs": n_epoch,
            "scheduler": scheduler,
            "decay_factor": decay_factor
        }
    )

    # 3- TRAINING
    trainer.train()
    metrics = trainer.get_metrics()
    fig = plot_metrics(metrics)
    fig.savefig(f'metrics_{run.name}.png', dpi=500, facecolor='white', edgecolor='none')

    # 4- TESTING
    trainer.test()
    # image_indices = random.sample(range(len(trainer.test_dataset)), 10)
    # for i in image_indices:
    #     image = trainer.test_dataset[i][0]
    #     mask = trainer.test_dataset[i][1]
    #     im = image.to(device)
    #     pred = model(im.unsqueeze(0))
    #     pred = pred.squeeze()
    #
    #     plot_subplots(im, mask, pred, trainer.imagenet_mean, trainer.imagenet_std)
