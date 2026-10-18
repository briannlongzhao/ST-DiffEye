# ST-DiffEye

Code for **ST-DiffEye: Diffusion-based Continuous Gaze Generation via Joint Scanpath-Trajectory Modeling**.

NeurIPS 2026

[arXiv](https://arxiv.org/abs/2606.15486) · [Project Page](https://st-diffeye.github.io/)

TLDR: ST-DiffEye is a diffusion model that generates continuous eye-trajectoy and aligned scanpath conditioned on a visual stimuli and an optional search target, using a per-point fixation that ties each trajectory point to human scanpath.


## Installation

The code was tested with Python 3.11, PyTorch 2.3.1 and CUDA 12.1 on a single GPU.

**Conda**

```bash
conda env create -f environment.yml
conda activate stdiffeye
```

**pip** (into an existing Python 3.11 environment)

```bash
pip install -r requirements.txt
```

## Checkpoints

Download the released checkpoints from Hugging Face into `checkpoints/`:

```bash
hf download briannlongzhao/ST-DiffEye --local-dir checkpoints
```

## Data

We use [COCO-Search18](https://sites.google.com/view/cocosearch/), [COCO-FreeView](https://sites.google.com/view/cocosearch/coco-freeview) and [MIT1003](https://people.csail.mit.edu/tjudd/WherePeopleLook/index.html).

```bash
# Scanpaths, images and image features from the ScanDiff data release
wget https://ailb-web.ing.unimore.it/publicfiles/ScanDiff_ICCV2025/data.zip
unzip data.zip -d data && rm data.zip

# MIT1003 gaze recordings
wget https://people.csail.mit.edu/tjudd/WherePeopleLook/DATA.zip
unzip DATA.zip -d data/mit1003 && rm DATA.zip
```

For COCO-Search18 and COCO-FreeView, request the raw EyeLink `.edf` gaze recordings from the dataset authors, put them under `data/cocosearch18/edfs/` and `data/coco_freeview/edfs/`, and convert them to `.asc` with `edf2asc` from the [EyeLink Developers Kit](https://www.sr-research.com/support/thread-13.html):

```bash
find data/cocosearch18/edfs data/coco_freeview/edfs -name "*.edf" -exec edf2asc -input {} \;
```

## Training and evaluation

Every dataset has an experiment preset in `configs/experiment/` that holds all of its settings, so each command only needs the preset name, the data folder and, for evaluation, a checkpoint. `src/train.py` trains a model; `src/eval.py` generates trajectories and scanpaths for the test set with a checkpoint and computes the metrics.

The default training batch sizes (128, or 64 for COCO-FreeView) need more than 24 GB of GPU memory. On smaller GPUs, lower it with `data.batch_size_train=...` (e.g. 32 for the COCO-Search18 and MIT1003 presets, 8 for COCO-FreeView on a 24 GB GPU).

### COCO-Search18, target present

```bash
# Train
python src/train.py experiment=cocosearch18_tp data_dir=/path/to/data
# Evaluate the released checkpoint
python src/eval.py experiment=cocosearch18_tp data_dir=/path/to/data ckpt_path=checkpoints/COCOSearch18-TP.pt
```

### COCO-Search18, target absent

```bash
# Train
python src/train.py experiment=cocosearch18_ta data_dir=/path/to/data
# Evaluate the released checkpoint
python src/eval.py experiment=cocosearch18_ta data_dir=/path/to/data ckpt_path=checkpoints/COCOSearch18-TA.pt
```

### COCO-FreeView

```bash
# Train
python src/train.py experiment=cocofreeview data_dir=/path/to/data
# Evaluate the released checkpoint
python src/eval.py experiment=cocofreeview data_dir=/path/to/data ckpt_path=checkpoints/COCOFreeView.pt
```

### MIT1003

```bash
# Train
python src/train.py experiment=mit1003 data_dir=/path/to/data
# Evaluate the released checkpoint
python src/eval.py experiment=mit1003 data_dir=/path/to/data ckpt_path=checkpoints/MIT1003.pt
```

## Demo

`src/demo.py` generates trajectories and scanpaths for a single image with a released checkpoint, and saves a figure with the trajectories (left) and scanpaths (right) overlaid on the image. It needs the checkpoints and `data/task_embeddings.npy` from the data release.

```bash
# Visual search: target present / target absent
python src/demo.py --image path/to/image.jpg --task cocosearch18_tp --target clock
python src/demo.py --image path/to/image.jpg --task cocosearch18_ta --target "potted plant"
# Free viewing
python src/demo.py --image path/to/image.jpg --task cocofreeview
python src/demo.py --image path/to/image.jpg --task mit1003
```

- `--task`: `cocosearch18_tp`, `cocosearch18_ta`, `cocofreeview` or `mit1003`; selects the checkpoint in `checkpoints/` (override with `--ckpt`).
- `--target`: search target for `cocosearch18_tp`/`cocosearch18_ta`, one of `bottle`, `bowl`, `car`, `chair`, `clock`, `cup`, `fork`, `keyboard`, `knife`, `laptop`, `microwave`, `mouse`, `oven`, `potted plant`, `sink`, `stop sign`, `toilet`, `tv`.
- `--num_samples`: number of generated samples (default 10); `--seed`: random seed for reproducible samples (default: none, so every run differs).
- `--out`: output figure (default `<image name>_<task>[_<target>].png` in the current folder).

## Citation

```bibtex
@article{zhao2026stdiffeye,
  title   = {ST-DiffEye: Diffusion-based Continuous Gaze Generation via Joint Scanpath-Trajectory Modeling},
  author  = {Zhao, Brian Nlong and Kara, Ozgur and Kim, Junho and Rehg, James M.},
  journal = {arXiv preprint arXiv:2606.15486},
  year    = {2026}
}
```

## Acknowledgments

This code builds on [ScanDiff](https://github.com/aimagelab/ScanDiff) and [DiffEye](https://github.com/rehg-lab/DiffEye).
