import fire
import skvideo.io
import face_recognition
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from os import listdir
from os.path import isfile, isdir, join
from math import floor, ceil


def read_video(filepath):
    return skvideo.io.vread(filepath)


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


def face_align_video(filepath, target_width, target_height):
    video_frames = read_video(filepath)
    aligned_frames = []

    for frame in video_frames:
        frame = extract_face(frame, target_width, target_height)
        aligned_frames.append(frame)

    write_video(filepath.replace('.mpg', '.aligned.mpg'), np.array(aligned_frames))


def file_listing(dir, extension=None):
    files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    if extension:
        files = list(filter(lambda f: f.endswith('.' + extension), files))
    return files


def dir_listing(base_dir):
    return [join(base_dir, d) for d in listdir(base_dir) if isdir(join(base_dir, d))]


def main(videos_path,
         target_width=156,
         target_height=156,
         num_cores=multiprocessing.cpu_count()):
    # Read all *.mpg filespaths
    filepaths = [f for d in dir_listing(videos_path) for f in file_listing(d, extension='mpg')]
    filepaths = tqdm(filepaths)
    # Align and save videos in parallel
    _ = Parallel(n_jobs=num_cores)(delayed(face_align_video)(p, target_width, target_height) for p in filepaths)


if __name__ == '__main__':
    fire.Fire(main)
