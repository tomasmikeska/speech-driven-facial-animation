# Speech-Driven Facial Animation

This project experiments with deep-learning approaches to convert voice recording to photorealistic and coherent video of
specific person synchronized with the input signal.

### Requirements

- Python 3.7 or newer
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


[TBD]
