# Convert LAFAN1 BVH sequences to SMPL-X format

## Introduction

This repository contains a script to convert the [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) BVH files to [SMPL-X](https://smpl-x.is.tue.mpg.de/) format. Fortunately, there is a one-to-one mapping between SMPL-X joints and the LAFAN1 skeleton, which makes the process fairly simple.

The SMPL-X shape coefficients (beta values) were computed to scale the SMPL-X model’s bones so that they match the BVH skeleton as closely as possible. After that, the script computes the local rotations of each BVH bone, and applies them to the SMPL-X model.

This script depends on [General Motion Retargeting](https://github.com/YanjieZe/GMR). Installing it should install all the other dependencies. Follow the instructions there to get the SMPL-X body models as well.

## Usage

```
python lafan_to_smplx.py --bvh_file [file from LAFAN1] --smplx_model_path [path to the folder containing the "smplx" folder with the SMPL-X body models] [--rerun]
```
