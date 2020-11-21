import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class AudioEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, 256)

    def forward(self, input_mfcc):
        emb = self.conv_net(input_mfcc.unsqueeze(1)).view(-1, 512)
        return torch.tanh(self.fc(emb))


class UNetFusion(pl.LightningModule):

    def __init__(self, audio_encoder):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.loss_fn = nn.L1Loss()
        self.learning_rate = 2e-4

        self.dconv_down1 = self.double_conv(3, 64)
        self.dconv_down2 = self.double_conv(64, 128)
        self.dconv_down3 = self.double_conv(128, 256)
        self.dconv_down4 = self.double_conv(256, 512)

        self.fus_conv = nn.Conv2d(512, 256, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = self.double_conv(256 + 512, 256)
        self.dconv_up2 = self.double_conv(128 + 256, 128)
        self.dconv_up1 = self.double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 3, 1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_in):
        conv1 = self.dconv_down1(x_in['still_image'])
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = F.relu(self.fus_conv(x))
        _, _, h, w = x.shape
        audio_emb = self.audio_encoder(x_in['audio'])
        audio_emb = audio_emb.view(-1, 256, 1, 1).repeat(1, 1, h, w)
        x = torch.cat([x, audio_emb], dim=1)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

    def training_step(self, x, batch_idx):
        x_hat = self.forward(x)
        loss = self.loss_fn(x_hat, x['frame'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, x, batch_idx):
        x_hat = self.forward(x)
        loss = self.loss_fn(x_hat, x['frame'])
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
