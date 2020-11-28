import sys
from os.path import dirname, join
sys.path.append(join(dirname(__file__), '../src/'))
import os
import fire
import pickle
import random
import numpy as np
import skvideo.io
import multiprocessing
from joblib import Parallel, delayed
from scipy.io import wavfile
from PIL import Image
from python_speech_features import mfcc
from tqdm import tqdm
from math import ceil
from extract_face import extract_face
from utils import mkdir, file_listing, dir_listing, get_file_name, file_exists


def read_video(filepath):
    return skvideo.io.vread(filepath)


def find_data_paths(base_path):
    audio_paths = [f for d in dir_listing(base_path + 'audio/') for f in file_listing(d, extension='wav')]
    landmark_paths = [f'{base_path}/landmarks/{get_file_name(p)}.npy' for p in audio_paths]
    video_paths = [p.replace('audio/', 'video/').replace('.wav', '.mpg') for p in audio_paths]
    paths = [{'audio': a, 'video': v, 'landmarks': l} for a, v, l in zip(audio_paths, video_paths, landmark_paths)]
    paths = [p for p in paths if file_exists(p['landmarks'])]
    return paths


def process_pair(pair_idx,
                 pair,
                 target_path,
                 audio_window_len,
                 mfcc_winlen,
                 mfcc_winstep,
                 mfcc_n,
                 img_width,
                 img_height,
                 num_still_images):
    audio_freq, audio_np = wavfile.read(pair['audio'])
    video_np = read_video(pair['video'])
    landmarks = np.load(pair['landmarks'], allow_pickle=True)
    video_frames = [extract_face(f, l, img_width, img_height) for f, l in zip(video_np, landmarks)]

    if len(video_frames) <= num_still_images:
        return

    video_meta = skvideo.io.ffprobe(pair['video'])
    video_frame_rate = int(video_meta['video']['@avg_frame_rate'].split('/')[0])  # Hz
    video_padding = ceil(audio_window_len / 2 * video_frame_rate)  # num of frames

    still_images = np.array(random.sample(video_frames, num_still_images))
    still_images_path = f'{target_path}/still_images/{pair_idx}.npy'
    np.save(still_images_path, still_images)

    for idx, frame in enumerate(video_frames[:-video_padding]):
        if idx > video_padding + 1:  # + 1 for still image (first frame in video)
            frame_time = idx / video_frame_rate  # seconds
            window_start_time = frame_time - audio_window_len / 2  # seconds
            window_start_idx = ceil(window_start_time * audio_freq)
            window_end_idx = window_start_idx + int(audio_window_len * audio_freq)

            audio_mfcc = mfcc(audio_np[window_start_idx:window_end_idx], audio_freq,
                              winlen=mfcc_winlen,
                              winstep=mfcc_winstep,
                              numcep=mfcc_n)

            frame_path = f'{target_path}/frames/{pair_idx}_{idx - video_padding}.jpg'
            Image.fromarray(frame).save(frame_path)

            data_point = {
                'frame_path': os.path.abspath(frame_path),
                'mfcc': audio_mfcc,
                'still_images_path': os.path.abspath(still_images_path)
            }
            pickle.dump(data_point, open(f'{target_path}/meta/{pair_idx}_{idx - video_padding}.pkl', 'wb'))


def build_dataset(base_path='data/grid/',
                  target_path='data/grid/processed/',
                  audio_win_len=0.35,
                  mfcc_winlen=0.025,
                  mfcc_winstep=0.01,
                  mfcc_n=13,
                  img_width=156,
                  img_height=156,
                  num_still_images=5,
                  num_cores=multiprocessing.cpu_count()):

    mkdir(target_path)
    mkdir(target_path + 'meta/')
    mkdir(target_path + 'frames/')
    mkdir(target_path + 'still_images/')

    pairs = find_data_paths(base_path)
    pairs = tqdm(pairs)

    _ = Parallel(n_jobs=num_cores)(
        delayed(process_pair)(idx, p, target_path, audio_win_len, mfcc_winlen, mfcc_winstep, mfcc_n,
                              img_width, img_height, num_still_images) for idx, p in enumerate(pairs))


if __name__ == '__main__':
    fire.Fire(build_dataset)
