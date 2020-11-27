# Speech-Driven Facial Animation

This project experiments with deep-learning approaches to convert voice recording to photorealistic and coherent video of
specific person synchronized with the input signal.

### Requirements

- Python 3.6+
- pip
- virtualenv
- ffmpeg

### Installation

Create and activate virtual environment
```
$ virtualenv venv
$ source venv/bin/activate
```

Install PyPI packages
```
$ pip install -r requirements.txt
```

Download GRID dataset
```
$ ./scripts/download-grid.sh
```

Extract face landmarks from all video frames \
*Note: This may take tens of GPU hours*
```
$ python scripts/extract_landmarks.py
```

Prepare dataset (extract faces from videos and create .pkl file for each data point including links to audio and frames)
```
$ python scripts/prepare_dataset.py
```

[optional] Add `.env` file with credentials configuration
```
COMET_API_KEY={apikey}
COMET_PROJECTNAME={projectname}
COMET_WORKSPACE={workspace}
```

### Usage

Training pipeline is configured using Hydra config files present in `configs/`. All options in config file
can be overwritten using command-line arguments. (Hydra docs: https://hydra.cc/docs/intro)

Train model
```
$ python src/train.py
```

Use trained model
```
$ python src/inference.py \
    --video_path {audio_input_file} \
    --checkpoint_path {trained_model_checkpoint}
```
