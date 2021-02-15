import os
import hydra
import multiprocessing
import pytorch_lightning as pl
from dotenv import load_dotenv
from hydra.utils import instantiate, to_absolute_path
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import GridDataset
from networks.baseline import UNetFusion
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def find_lr(trainer, model, dataloader):
    lr_finder = trainer.tuner.lr_find(model, train_dataloader=dataloader)
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr_find_fig.png')
    print('Suggested_lr:', lr_finder.suggestion())


def load_dataset(cfg):
    dataset = GridDataset(to_absolute_path(cfg.dataset_path),
                          n_identity_images=cfg.num_still_images,
                          id_image_transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize((cfg.input_image_width, cfg.input_image_height)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ]),
                          target_image_transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize((cfg.output_image_width, cfg.output_image_height)),
                              transforms.ToTensor()
                          ]))
    train_size = int(cfg.train_split_size * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


def get_data_loaders(cfg):
    num_workers = cfg.data_loader_workers or multiprocessing.cpu_count()
    train, val = load_dataset(cfg)
    train_loader = DataLoader(train, batch_size=cfg.train_batch_size, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=cfg.val_batch_size, num_workers=num_workers)
    return train_loader, val_loader


def load_comet_logger(cfg):
    return CometLogger(api_key=os.getenv('COMET_API_KEY'),
                       project_name=os.getenv('COMET_PROJECTNAME'),
                       workspace=os.getenv('COMET_WORKSPACE'),
                       save_dir=to_absolute_path(cfg.logs.path))


@hydra.main(config_path='../configs/', config_name='baseline')
def train(cfg):
    train_loader, val_loader = get_data_loaders(cfg)
    if cfg.model.ckpt_path:
        model = UNetFusion.load_from_checkpoint(to_absolute_path(cfg.model.ckpt_path))
    else:
        model = instantiate(cfg.model)

    comet_logger = load_comet_logger(cfg) if cfg.logs.use_comet else None
    logger = comet_logger or TensorBoardLogger(to_absolute_path(cfg.logs.path))

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        filename=f'{model}' + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=cfg.checkpoint.save_top_k,
        mode=cfg.checkpoint.mode
    )

    early_stop_callback = EarlyStopping(
        monitor=cfg.early_stopping_monitor,
        min_delta=cfg.early_stopping_delta,
        patience=cfg.early_stopping_patience,
        mode=cfg.early_stopping_mode
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        val_check_interval=cfg.val_check_interval,
        num_sanity_val_steps=-1,
        gpus=cfg.gpus,
        precision=cfg.precision,
        terminate_on_nan=True,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accumulate_grad_batches=4
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    load_dotenv()
    train()
