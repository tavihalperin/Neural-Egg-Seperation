# Neural Egg Seperation

This repository contains demo code for the paper <a href = "https://arxiv.org/abs/1811.12739"> Neural separation of observed and unobserved distributions </a> (ICML 2019).

Please cite the papers below if you make use of the software.

## Prerequisites
The following packages are required:
```
torch
torchvision
numpy 
museval (for audio)
```

## Usage

First, download speech and shoes/bags training sets from <a href = "https://drive.google.com/drive/folders/1TLDV1rQhUGpYDe48RYsQzP5SdbVRgRZV?usp=sharing"> Google drive </a>
Store speech train data in ```data/speech```, bags in ```data/bags```, and shoes in ```data/shoes```. 

Run
```bash
python train_audio.py
```

Or
```bash
python train_images.py
```


## Publications

```
@article{halperin2018neural,
  title={Neural separation of observed and unobserved distributions},
  author={Halperin, Tavi and Ephrat, Ariel and Hoshen, Yedid},
  journal={arXiv preprint arXiv:1811.12739},
  year={2018}
}
```
