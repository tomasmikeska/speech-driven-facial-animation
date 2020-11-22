import os
import fire
import torch
import numpy as np
import ffmpeg
from PIL import Image
from scipy.io import wavfile
from torchvision import transforms
from python_speech_features import mfcc
from networks.baseline import UNetFusion


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


def main(audio_path, still_image_path, checkpoint_path,
         output_path='generated.mp4',
         output_freq=25,
         input_img_width=96,
         input_img_height=96,
         audio_window_size=0.35,
         mfcc_winlen=0.025,
         mfcc_winstep=0.01,
         mfcc_n=13):

    img_transform = transforms.Compose([
        transforms.Resize((input_img_height, input_img_width)),
        transforms.ToTensor()
    ])
    # Load model
    model = UNetFusion.load_from_checkpoint(checkpoint_path)
    # Read image
    still_img = Image.open(still_image_path)
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

        still_image_tensor = img_transform(still_img).unsqueeze(0)
        audio_frame_tensor = torch.from_numpy(audio_frame_mfcc).unsqueeze(0)

        output_image = model({'audio': audio_frame_tensor, 'still_image': still_image_tensor})
        output_image = output_image[0].cpu().detach().numpy().transpose((1, -1, 0))
        images.append(output_image)

    video_np = np.array(images) * 255  # Create (0, 255) range numpy array representing video
    write_video(output_path, video_np.astype(np.uint8))  # Write video without sound
    os.system(f'ffmpeg -i {output_path} -i {audio_path} -c:v copy -c:a aac -y {output_path}')  # Merge sound with video


if __name__ == '__main__':
    fire.Fire(main)
