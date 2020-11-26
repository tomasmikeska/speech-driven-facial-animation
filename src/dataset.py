import pickle
import torch
import numpy as np
from matplotlib.pyplot import imread
from torch.utils.data import Dataset
from utils import file_listing


class GridDataset(Dataset):

    def __init__(self, base_path,
                 audio_transform=None,
                 id_image_transform=None,
                 target_image_transform=None):
        self.base_path = base_path
        self.filepaths = file_listing(base_path + '/meta/', extension='pkl')
        self.audio_transform = audio_transform
        self.id_image_transform = id_image_transform
        self.target_image_transform = target_image_transform

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        frame_data = pickle.load(open(self.filepaths[index], 'rb'))
        frame = imread(frame_data['frame_path'])
        still_images = np.load(frame_data['still_images_path'])
        audio_mfcc = frame_data['mfcc'].astype('float32')
        # Normalize MFCC features
        audio_mfcc = audio_mfcc - audio_mfcc.mean()
        audio_scale = np.absolute(audio_mfcc).max()
        audio_mfcc = audio_mfcc / (audio_scale if audio_scale != 0 else 1)

        if self.audio_transform:
            audio_mfcc = self.audio_transform(audio_mfcc)
        if self.id_image_transform:
            still_images = [self.id_image_transform(img) for img in still_images]
        if self.target_image_transform:
            frame = self.target_image_transform(frame)

        return {'audio': audio_mfcc, 'frame': frame, 'still_images': still_images}

    def __len__(self):
        return len(self.filepaths)
