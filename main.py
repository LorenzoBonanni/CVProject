import copy
import logging
import os
import random

import numpy as np
import torch
import wandb
from monai.losses import DiceCELoss

from models.mae_unetr import MaeUnetr
from trainer import Trainer
from utilis.utils import plot_metrics, get_opt, plot_subplots, get_model

SEED = 5
LOGGER = logging.getLogger(__name__)
use_cuda = torch.cuda.is_available()


def seed_everything(seed: int):
    """
    Seed various random number generators for reproducibility.

    This function sets seeds for random number generators in Python, PyTorch, NumPy,
    and the operating system environment variables to ensure reproducibility of results.

    :param seed: The seed value to be used for random number generation.
    :type seed: int
    """
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


def main(opt):
    device = torch.device(f'cuda:{opt.device}' if use_cuda else 'cpu')
    n_epoch, lr, scheduler, decay_factor = opt.epochs, opt.lr, opt.scheduler, opt.decay_factor
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        handlers=[
            # logging.FileHandler(f'{base_path}/{opt["numberepochs"]}/INFO.log', 'w'),
            logging.StreamHandler()
        ]

    )

    # 1- MODEL
    model = get_model(opt.model_name, opt)
    model.to(device)
    # 2- DATA LOADING AND TRAINER
    trainer = Trainer(
        model=model,
        batch_size=opt.batch_size,
        device=device,
        num_epochs=n_epoch,
        optimizer=torch.optim.Adam(lr=lr, params=model.get_parameters()),
        criterion=DiceCELoss(to_onehot_y=True,
                             softmax=True,
                             squared_pred=False),
        dataset_name=opt.dataset,
        dataset_path=opt.dataset_path,
        scheduler=scheduler,
        decay_factor=decay_factor,
        start_lr=copy.copy(lr)
    )
    run = wandb.init(
        project="UnetMae",
        config=opt.__dict__,
        # tags=["final_exp"]
    )

    # 3- TRAINING
    # Mae
    start_epoch = 0
    if opt.train_mae:
        LOGGER.info(f"Training All the Network")
        trainer2 = Trainer(
            model=model,
            batch_size=16,
            device=device,
            num_epochs=opt.mae_epochs,
            optimizer=torch.optim.Adam(lr=opt.mae_lr, params=model.get_parameters()),
            criterion=DiceCELoss(to_onehot_y=True,
                                 softmax=True,
                                 squared_pred=False),
            dataset_name=opt.dataset,
            dataset_path=opt.dataset_path,
            scheduler=False,
            decay_factor=decay_factor,
            start_lr=copy.copy(lr)
        )
        trainer2.train()
        model.train_mae = False
        metrics2 = trainer2.get_metrics()
        start_epoch = opt.mae_epochs
        fig2 = plot_metrics(metrics2)
        fig2.savefig(f'metrics2_{run.name}.png', dpi=500, facecolor='white', edgecolor='none')

    if isinstance(model, MaeUnetr):
        model.freeze_mae()
    trainer.train(start_epoch)
    trainer.validate(start_epoch+n_epoch)
    trainer.val_epochs.append(n_epoch)
    metrics1 = trainer.get_metrics()
    fig1 = plot_metrics(metrics1)
    fig1.savefig(f'metrics1_{run.name}.png', dpi=500, facecolor='white', edgecolor='none')

    LOGGER.info(f"Best Epoch: {opt.mae_epochs+trainer.best_epoch}")

    # 4- TESTING

    trainer.test()
    random.seed(0)
    image_indices = random.sample(range(len(trainer.test_dataset)), 10)
    for i in image_indices:
        image = trainer.test_dataset[i][0]
        mask = trainer.test_dataset[i][1]
        im = image.to(device)
        pred = model(im.unsqueeze(0))
        pred = pred.squeeze()

        final_fig = plot_subplots(im, mask, pred, trainer.imagenet_mean, trainer.imagenet_std)
        wandb.log({f"fig": wandb.Image(final_fig)})


if __name__ == '__main__':
    seed_everything(SEED)
    opt = get_opt()
    for exp_num in range(opt.num_exp):
        main(opt)
