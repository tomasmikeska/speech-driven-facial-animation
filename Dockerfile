FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# --------------------------------------------------------------------
# System libs
# --------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential apt-utils ca-certificates wget git

# --------------------------------------------------------------------
# OpenCV
# --------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake unzip pkg-config \
        libjpeg-dev libpng-dev libtiff-dev \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev \
        libatlas-base-dev gfortran \
        libsm6 libxext6 libxrender-dev

# --------------------------------------------------------------------
# FFmpeg
# --------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg

# --------------------------------------------------------------------
# Copy files and set workdir
# --------------------------------------------------------------------

COPY . /speech-driven-facial-animation/
WORKDIR /speech-driven-facial-animation

# --------------------------------------------------------------------
# PyPI packages
# --------------------------------------------------------------------

RUN pip install -r requirements.txt

# --------------------------------------------------------------------
# Ports
# --------------------------------------------------------------------

# Tensorboard port
EXPOSE 6006
