import os
import fire
import pickle
import skvideo.io
import numpy as np
import multiprocessing
import face_recognition
from joblib import Parallel, delayed
from scipy.io import wavfile
from os import listdir
from os.path import isfile, isdir, join
from tqdm import tqdm
from math import floor, ceil


STILL_IMAGE_OFFSET = 3


def write_video(filepath, frames):
    skvideo.io.vwrite(filepath, frames.astype(np.uint8))


def extract_face(frame, target_width, target_height):
    faces = face_recognition.face_locations(frame)
    if len(faces) != 1:
        return frame
    top, right, bottom, left = faces[0]
    width = right - left
    height = bottom - top
    w_padding = (target_width - width) / 2
    h_padding = (target_height - height) / 2
    return frame[top - ceil(h_padding):bottom + floor(h_padding),
                 left - ceil(w_padding):right + floor(w_padding)]


def read_audio(filepath):
    return wavfile.read(filepath)


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


def process_pair(pair_idx, pair, target_path, audio_window_len, img_width, img_height):
    audio_freq, audio_np = read_audio(pair['audio'])
    video_np = read_video(pair['video'])

    video_meta = skvideo.io.ffprobe(pair['video'])
    video_frame_rate = int(video_meta['video']['@avg_frame_rate'].split('/')[0])  # Hz
    video_padding = ceil(audio_window_len / 2 * video_frame_rate)  # num of frames
    still_image = video_np[STILL_IMAGE_OFFSET]
    still_image = extract_face(still_image, img_width, img_height)

    for idx, frame in enumerate(video_np[:-video_padding]):
        frame = extract_face(frame, img_width, img_height)
        if idx > video_padding + 1:  # + 1 for still image (first frame in video)
            frame_time = idx / video_frame_rate  # seconds
            window_start_time = frame_time - audio_window_len / 2  # seconds
            window_start_idx = ceil(window_start_time * audio_freq)
            window_end_idx = window_start_idx + int(audio_window_len * audio_freq)
            audio_slice = audio_np[window_start_idx:window_end_idx]

            data_point = {'frame': frame, 'audio': audio_slice, 'still_image': still_image}
            pickle.dump(data_point, open(f'{target_path}/grid_{pair_idx}_{idx - video_padding}.pkl', 'wb'))


def build_dataset(base_path='data/grid/',
                  target_path='data/grid/processed/',
                  audio_win_len=0.35,
                  img_width=156,
                  img_height=156,
                  num_cores=multiprocessing.cpu_count()):

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    pairs = find_data_paths(base_path)
    pairs = tqdm(pairs)

    _ = Parallel(n_jobs=num_cores)(
        delayed(process_pair)(idx, p, target_path, audio_win_len, img_width, img_height) for idx, p in enumerate(pairs))


if __name__ == '__main__':
    fire.Fire(build_dataset)
