import pickle
import torch
import random
import numpy as np
import skvideo.io
from matplotlib.pyplot import imread
from torch.utils.data import Dataset
from utils import file_listing
from extract_face import extract_face


class GridDataset(Dataset):

    def __init__(self, base_path,
                 n_identity_images=16,
                 audio_transform=None,
                 id_image_transform=None,
                 target_image_transform=None):
        self.base_path = base_path
        self.n_identity_images = n_identity_images
        self.filepaths = file_listing(base_path + '/meta/', extension='pkl')
        self.audio_transform = audio_transform
        self.id_image_transform = id_image_transform
        self.target_image_transform = target_image_transform
        self.identity_video_cache = {}

    def _read_video(self, path, landmarks_path):
        if path in self.identity_video_cache:
            return self.identity_video_cache[path]
        else:
            video_np = skvideo.io.vread(path)
            video_landmarks = np.load(landmarks_path, allow_pickle=True)
            video_frames = [extract_face(f, l) for f, l in zip(video_np, video_landmarks)]
            if self.id_image_transform:
                video_frames = [self.id_image_transform(frame) for frame in video_frames]
            self.identity_video_cache[path] = video_frames
            return video_frames

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        frame_data = pickle.load(open(self.filepaths[index], 'rb'))
        frame = imread(frame_data['frame_path'])
        identity_video = self._read_video(frame_data['identity_video'], frame_data['identity_video_landmarks'])
        still_images = random.sample(list(identity_video), self.n_identity_images)
        audio_mfcc = frame_data['mfcc'].astype('float32')
        # Normalize MFCC features
        audio_mfcc = audio_mfcc - audio_mfcc.mean()
        audio_scale = np.absolute(audio_mfcc).max()
        audio_mfcc = audio_mfcc / (audio_scale if audio_scale != 0 else 1)

        if self.audio_transform:
            audio_mfcc = self.audio_transform(audio_mfcc)
        if self.target_image_transform:
            frame = self.target_image_transform(frame)

        return {'audio': audio_mfcc, 'frame': frame, 'still_images': still_images}

    def __len__(self):
        return len(self.filepaths)
