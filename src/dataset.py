import skvideo.io
import python_speech_features
from scipy.io import wavfile
from sklearn import preprocessing


MFCC_SLIDING_WINDOW_SIZE = 0.025  # seconds
MFCC_SLIDING_WINDOW_STEP = 0.01  # seconds
MFCC_COEFS = 13


def normalize(x):
    return (x - x.mean()) / (x.max() - x.min())


def mfcc(audio, freq):
    mfcc_data = python_speech_features.mfcc(audio, freq,
                                            winlen=MFCC_SLIDING_WINDOW_SIZE,
                                            winstep=MFCC_SLIDING_WINDOW_STEP,
                                            numcep=MFCC_COEFS,
                                            appendEnergy=True)
    return preprocessing.scale(mfcc_data)


def extract_audio_features(audio, freq):
    audio = normalize(audio)
    return mfcc(audio, freq)


def read_wav(filepath):
    freq, audio = wavfile.read(filepath)
    return extract_audio_features(audio, freq)


def read_video(filepath):
    return skvideo.io.vread(filepath)
