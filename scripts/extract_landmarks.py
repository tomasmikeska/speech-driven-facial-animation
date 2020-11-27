import sys
from os.path import dirname, join
sys.path.append(join(dirname(__file__), '../src/'))
import fire
import torch
import skvideo.io
import numpy as np
import face_alignment
from tqdm import tqdm
from utils import mkdir, file_listing, dir_listing, get_file_name


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='blazeface')


def extract_landmarks(dataset_path='data/grid/', initial_index=0):
    mkdir(dataset_path + '/landmarks')
    filepaths = [f for d in dir_listing(dataset_path + '/video/') for f in file_listing(d, extension='mpg')]
    filepaths_subset = filepaths[initial_index:]

    for video_path in tqdm(filepaths_subset, initial=initial_index, total=len(filepaths)):
        frames = skvideo.io.vread(video_path).transpose(0, 3, 1, 2)
        video_tensor = torch.Tensor(frames)
        landmarks = fa.get_landmarks_from_batch(video_tensor.cuda())
        np.save(f'{dataset_path}/landmarks/{get_file_name(video_path)}.npy', landmarks)


if __name__ == '__main__':
    fire.Fire(extract_landmarks)
