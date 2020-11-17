# Speech-Driven Facial Animation

This project experiments with deep-learning approaches to convert voice recording to photorealistic and coherent video of
specific person synchronized with the input signal.

### Requirements

- Python 3.7 or newer
- pip
- virtualenv

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

Extract faces from videos (creates videos with same paths only with '.aligned.mpg' extension)
```
$ python scripts/align_videos.py --videos_path 'data/grid/video/'
```


[TBD]
