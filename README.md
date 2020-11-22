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

Prepare dataset (create .pkl files with short audio recordings with corresponding input and output image frame with faces extracted)
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

Training pipeline is configures using Hydra configuration files present in `configs/`. All options in used config file
can be overwritten using command-line arguments. (Hydra docs: https://hydra.cc/docs/intro)

Train model
```
$ python src/train.py
```

Use trained model
```
$ python src/inference.py \
    --audio_path {audio_input_file} \
    --still_image_path {identity_image} \
    --checkpoint_path {trained_model_checkpoint}
```
