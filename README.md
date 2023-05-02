# SGARN
![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)
![PyTorch >=1.1](https://img.shields.io/badge/PyTorch->=1.1-blue.svg)

# Structure-embedded Ghosting Artifact Suppression Network for High Dynamic Range Image Reconstruction

The *official* repository for  [Structure-embedded Ghosting Artifact Suppression Network for High Dynamic Range Image Reconstruction].

## Pipeline

![framework](/SGARN/fig/framework.png)

## Requirements

### Installation

```bash
we use /torch >=1.1 / 24G  RTX3090 for training and evaluation.

```

### Prepare Datasets

```bash
mkdir data
```

Download the HDR datasets [Kalantari](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/).

Then unzip them and rename them under the directory like.

```
data
├── Training
│   └── 
│   └── 
│   └── 
├── Test
│   └── 
│   └── 
│   └── 
```

### Train
We utilize 1 RTX3090 GPU for training.



## Evaluation

```bash
python test.py
```

## Contact

If you have any questions, please feel free to contact me.(tanglf111@qq.com).


## Citation
```text
@article{
title = {Structure-embedded ghosting artifact suppression network for high dynamic range image reconstruction},
journal = {Knowledge-Based Systems},
volume = {263},
pages = {110278},
year = {2023},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2023.110278},
}
```
