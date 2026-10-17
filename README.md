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

Datasets are organized in the following directory.

Please consult the authors of COCO-Search18 and COCO-FreeView for access to raw trajectory data in EDF format.

```text
data/
├── task_embeddings.npy
├── cocosearch18_edf/
│   ├── coco_search18_fixations_TP_{train,validation,test}.json
│   ├── coco_search18_fixations_TA_{trainval,test}.json
│   ├── cocosearch18_edf_cache.pkl
│   ├── clusters_cocosearch18_edf_{tp,ta}_512_384.npy
│   ├── images/<category>/
│   ├── dinov2_base_timm_image_features_{tp,ta}/<category>/
│   └── SemSS/
├── cocofreeview_edf/
│   ├── COCOFreeView_fixations_{train,validation,test}.json
│   ├── cocofreeview_edf_cache.pkl
│   ├── clusters_coco_freeview_512_384.npy
│   ├── images/
│   └── dinov2_base_timm_image_features/
└── mit1003_edf/
    ├── mit1003_fixations_{train,validation,test}.json
    ├── mit1003_edf_cache.pkl
    ├── clusters_mit1003_512_384.npy
    └── dinov2_base_timm_image_features/
```

## Training and evaluation

Every dataset has an experiment preset in `configs/experiment/` that holds all of its settings, so each command only needs the preset name, the data folder and, for evaluation, a checkpoint. `src/train.py` trains a model; `src/eval.py` generates trajectories and scanpaths for the test set with a checkpoint and computes the metrics.

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
