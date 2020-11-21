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


class IdentityEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3)
        self.conv4 = nn.Conv2d(512, 512, 3)
        self.conv5 = nn.Conv2d(512, 512, 3)
        self.fc6 = nn.Linear(512, 256)

    def forward(self, input_image):
        conv1_out = F.relu(self.conv1(input_image))
        x = self.pool1(conv1_out)
        conv2_out = F.relu(self.conv2(x))
        x = self.pool2(conv2_out)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        emb = x.view(-1, 512)
        return torch.tanh(self.fc6(emb)), conv1_out, conv2_out


class Decoder(pl.LightningModule):

    def __init__(self, latent_size=512):
        super().__init__()
        self.latent_size = latent_size
        self.conv_t7 = nn.ConvTranspose2d(64, 3, 5)
        self.conv_t6 = nn.ConvTranspose2d(96, 64, 5, stride=2)
        self.conv_t5 = nn.ConvTranspose2d(96, 96, 5, stride=2)
        self.conv_t4 = nn.ConvTranspose2d(256, 96, 5, stride=2)
        self.conv_t3 = nn.ConvTranspose2d(512, 256, 5, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(256, 512, 6, stride=2)
        self.fc = nn.Linear(latent_size, 256)

    def forward(self, x, skip_conn_t4, skip_conn_t5):
        print('skip_conn_t4.shape', skip_conn_t4.shape)
        print('skip_conn_t5.shape', skip_conn_t5.shape)
        x = torch.tanh(self.fc(x))
        x = x.view(-1, 256, 1, 1)
        x = F.relu(self.conv_t2(x))
        x = F.relu(self.conv_t3(x))
        x = F.relu(self.conv_t4(x))
        print(x.shape)
        x = F.relu(self.conv_t5(x))
        print(x.shape)
        x = F.relu(self.conv_t6(x))
        return torch.tanh(self.conv_t7(x))


class Speech2Vid(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.identity_encoder = IdentityEncoder()
        self.decoder = Decoder()
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        audio_input = x['audio']
        image_input = x['id_image']

        audio_embedding = self.audio_encoder(audio_input)
        image_embedding, skip_conn_t5, skip_conn_t4 = self.identity_encoder(image_input)
        embedding = torch.cat((audio_embedding, image_embedding), 1)

        return self.decoder(embedding, skip_conn_t4, skip_conn_t5)

    def training_step(self, x, batch_idx):
        x_hat = self.forward(x)
        print('x_hat.shape', x_hat.shape)
        loss = self.loss_fn(x_hat, x['frame'])
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
