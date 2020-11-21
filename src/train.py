import os
import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from hydra.utils import instantiate, to_absolute_path
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import GridDataset
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def find_lr(trainer, model, dataloader):
    lr_finder = trainer.tuner.lr_find(model, train_dataloader=dataloader)
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr_find_fig.png')
    print('Suggested_lr:', lr_finder.suggestion())


def load_dataset(cfg):
    dataset = GridDataset(to_absolute_path(cfg.dataset_path),
                          cfg.audio_freq,
                          cfg.mfcc_winlen,
                          cfg.mfcc_winstep,
                          cfg.mfcc_n,
                          id_image_transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize((cfg.input_image_width, cfg.input_image_height)),
                              transforms.ToTensor()
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
    train, val = load_dataset(cfg)
    train_loader = DataLoader(train, batch_size=cfg.train_batch_size, num_workers=8)
    val_loader = DataLoader(val, batch_size=cfg.val_batch_size, num_workers=8)
    return train_loader, val_loader


def load_comet_logger(cfg):
    return CometLogger(api_key=os.getenv('COMET_API_KEY'),
                       project_name=os.getenv('COMET_PROJECTNAME'),
                       workspace=os.getenv('COMET_WORKSPACE'),
                       save_dir=to_absolute_path(cfg.logs.path))


@hydra.main(config_path='../configs/', config_name='baseline')
def train(cfg):
    train_loader, val_loader = get_data_loaders(cfg)
    model = instantiate(cfg.model)

    comet_logger = load_comet_logger(cfg) if cfg.logs.use_comet else None
    logger = comet_logger or TensorBoardLogger(to_absolute_path(cfg.logs.path))

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        filename=f'{model}' + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=cfg.checkpoint.save_top_k,
        mode=cfg.checkpoint.mode)

    trainer = pl.Trainer(gpus=cfg.gpus,
                         precision=cfg.precision,
                         fast_dev_run=True,
                         logger=logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    load_dotenv()
    train()


# Upravit prepare_dataset.py - paralelni a spojit s align
# Inferencni skript
# vyresit logovani val_loss
# logovat obrazky
# Refactoring
# Doprocesovat dataset
# Upravit README.md
