import pickle
import torch
from python_speech_features import mfcc
from torch.utils.data import Dataset
from utils import file_listing


class GridDataset(Dataset):

    def __init__(self, base_path,
                 audio_transform=None,
                 id_image_transform=None,
                 target_image_transform=None):
        self.base_path = base_path
        self.filepaths = file_listing(base_path, extension='pkl')
        self.audio_transform = audio_transform
        self.id_image_transform = id_image_transform
        self.target_image_transform = target_image_transform

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        data_point = pickle.load(open(self.filepaths[index], 'rb'))
        data_point['audio'] = mfcc(data_point['audio'], 16000,
                                   winlen=0.025,
                                   winstep=0.01,
                                   numcep=13).astype('float32')

        if self.audio_transform:
            data_point['audio'] = self.audio_transform(data_point['audio'])
        if self.id_image_transform:
            data_point['still_image'] = self.id_image_transform(data_point['still_image'])
        if self.target_image_transform:
            data_point['frame'] = self.target_image_transform(data_point['frame'])

        return data_point

    def __len__(self):
        return len(self.filepaths)
