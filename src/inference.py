import os
import fire
import torch
import random
import numpy as np
import ffmpeg
import skvideo.io
import face_alignment
from scipy.io import wavfile
from torchvision import transforms
from python_speech_features import mfcc
from networks.baseline import UNetFusion
from extract_face import extract_face, extract_landmarks


def get_audio_window(audio_np, audio_freq, start_time, end_time):
    start_index = round(start_time * audio_freq)
    padding_left = np.zeros(abs(min(start_index, 0)))
    end_index = int(end_time * audio_freq)
    return np.concatenate((padding_left, audio_np[max(start_index, 0):end_index]))


def write_video(output_path, images, framerate=25, vcodec='libx264'):
    n, height, width, channels = images.shape
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(output_path, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def main(video_path, checkpoint_path,
         output_path='generated.mp4',
         output_freq=25,
         num_still_images=16,
         img_width=256,
         img_height=256,
         audio_window_size=0.35,
         mfcc_winlen=0.025,
         mfcc_winstep=0.01,
         mfcc_n=13):

    audio_path = 'temp.wav'
    os.system(f'ffmpeg -i {video_path} -vn -ar 16000 -ac 1 -y {audio_path}')

    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load model
    model = UNetFusion.load_from_checkpoint(checkpoint_path)
    # Read video
    video_np = skvideo.io.vread(video_path)
    still_images = random.sample(list(video_np), num_still_images)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='blazeface')
    still_images = [extract_face(img, extract_landmarks(img, fa)[0], img_width, img_height) for img in still_images]
    still_images = [img_transform(img).unsqueeze(0) for img in still_images]
    # Read audio
    audio_freq, audio_np = wavfile.read(audio_path)

    images = []
    t = 0  # time of each frame

    while True:
        t += 1 / output_freq

        if (t + audio_window_size / 2) * audio_freq > len(audio_np):  # if end of audio was reached
            break

        audio_frame = get_audio_window(audio_np, audio_freq, t - audio_window_size / 2, t + audio_window_size / 2)
        audio_frame_mfcc = mfcc(audio_frame, audio_freq,
                                winlen=mfcc_winlen,
                                winstep=mfcc_winstep,
                                numcep=mfcc_n).astype('float32')
        # Normalize MFCC features
        audio_frame_mfcc = audio_frame_mfcc - audio_frame_mfcc.mean()
        audio_scale = np.absolute(audio_frame_mfcc).max()
        audio_frame_mfcc = audio_frame_mfcc / (audio_scale if audio_scale != 0 else 1)
        audio_frame_tensor = torch.from_numpy(audio_frame_mfcc).unsqueeze(0)

        output_image = model({'audio': audio_frame_tensor, 'still_images': still_images})
        output_image = output_image[0].cpu().detach().numpy().transpose((1, -1, 0))
        images.append(output_image)

    video_np = np.array(images) * 255  # Create (0, 255) range numpy array representing video
    write_video(output_path, video_np.astype(np.uint8))  # Write video without sound
    os.system(f'ffmpeg -i {output_path} -i {audio_path} -c:v copy -c:a aac -y {output_path}')  # Merge sound with video
    os.system(f'rm {audio_path}')


if __name__ == '__main__':
    fire.Fire(main)
