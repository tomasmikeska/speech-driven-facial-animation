import os
import fire
import pytorch_lightning as pl
from dotenv import load_dotenv
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import GridDataset
from networks.baseline import UNetFusion, AudioEncoder
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger


def find_lr(trainer, model, dataloader):
    lr_finder = trainer.tuner.lr_find(model, train_dataloader=dataloader)
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr_find_fig.png')
    print('Suggested_lr:', lr_finder.suggestion())


def train(dataset_path='data/processed/', img_width=96, img_height=96):
    dataset = GridDataset(dataset_path,
                          id_image_transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize((img_width, img_height)),
                              transforms.ToTensor()
                          ]),
                          target_image_transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize((img_width, img_height)),
                              transforms.ToTensor()
                          ]))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    if os.getenv('COMET_API_KEY'):
        comet_logger = CometLogger(api_key=os.getenv('COMET_API_KEY'),
                                   project_name=os.getenv('COMET_PROJECTNAME'),
                                   workspace=os.getenv('COMET_WORKSPACE'),
                                   save_dir='logs/')

    audio_encoder = AudioEncoder()
    unet = UNetFusion(audio_encoder)
    trainer = pl.Trainer(gpus=1,
                         precision=16,
                         logger=comet_logger or TensorBoardLogger('logs/'))
    trainer.fit(unet, DataLoader(train, batch_size=128, num_workers=8), DataLoader(val, batch_size=128))


if __name__ == '__main__':
    load_dotenv()
    fire.Fire(train)
