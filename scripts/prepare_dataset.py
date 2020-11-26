import os
import fire
import pickle
import random
import numpy as np
import skvideo.io
import multiprocessing
import face_recognition
from joblib import Parallel, delayed
from scipy.io import wavfile
from PIL import Image
from python_speech_features import mfcc
from os import listdir
from os.path import isfile, isdir, join
from tqdm import tqdm
from math import ceil


def extract_face(frame, target_width, target_height):
    faces = face_recognition.face_locations(frame)
    img_pil = Image.fromarray(frame)

    if len(faces) != 1:
        return np.array(img_pil.resize((target_width, target_height), Image.ANTIALIAS))

    top, right, bottom, left = faces[0]
    x, y, width, height = left, top, right - left, bottom - top

    if width / height > target_width / target_height:
        fin_width  = width
        fin_height = int(fin_width * (target_height / float(target_width)))
        fin_x      = x
        fin_y      = y - (fin_height - height) / 2
    else:
        fin_height = height
        fin_width  = int(fin_height * (target_width / float(target_height)))
        fin_x      = x - (fin_width - width) / 2
        fin_y      = y

    img_pil = img_pil.crop((fin_x, fin_y, fin_x + fin_width, fin_y + fin_height))
    img_pil = img_pil.resize((target_width, target_height), Image.ANTIALIAS)
    return np.array(img_pil)


def read_video(filepath):
    return skvideo.io.vread(filepath)


def file_listing(dir, extension=None):
    files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    if extension:
        files = list(filter(lambda f: f.endswith('.' + extension), files))
    return files


def dir_listing(base_dir):
    return [join(base_dir, d) for d in listdir(base_dir) if isdir(join(base_dir, d))]


def find_data_paths(base_path):
    audio_paths = [f for d in dir_listing(base_path + 'audio/') for f in file_listing(d, extension='wav')]
    audio_video_pairs = [(p, p.replace('audio/', 'video/').replace('.wav', '.mpg')) for p in audio_paths]
    return [{'audio': p[0], 'video': p[1]} for p in audio_video_pairs]


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
    video_frames = [extract_face(frame, img_width, img_height) for frame in video_np]

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

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if not os.path.exists(target_path + 'meta/'):
        os.makedirs(target_path + 'meta/')
    if not os.path.exists(target_path + 'frames/'):
        os.makedirs(target_path + 'frames/')
    if not os.path.exists(target_path + 'still_images/'):
        os.makedirs(target_path + 'still_images/')

    pairs = find_data_paths(base_path)
    pairs = tqdm(pairs)

    _ = Parallel(n_jobs=num_cores)(
        delayed(process_pair)(idx, p, target_path, audio_win_len, mfcc_winlen, mfcc_winstep, mfcc_n,
                              img_width, img_height, num_still_images) for idx, p in enumerate(pairs))


if __name__ == '__main__':
    fire.Fire(build_dataset)
