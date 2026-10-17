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

We use the following datasets:

- [COCO-Search18](https://sites.google.com/view/cocosearch/) (target present and target absent)
- [COCO-FreeView](https://sites.google.com/view/cocosearch/coco-freeview)
- [MIT1003](https://people.csail.mit.edu/tjudd/WherePeopleLook/index.html)

Each dataset combines two sources:

- **Scanpaths, images and features** from the [ScanDiff](https://github.com/aimagelab/ScanDiff) data release: fixation JSON files and splits, images, DINOv2 image features, the clusters and semantic maps used by the metrics, and the task embeddings.
- **Raw gaze recordings.** For COCO-Search18 and COCO-FreeView, these are the EyeLink EDF files, shared by the dataset authors on request; please contact them for access. For MIT1003, they are the eye-tracking `.mat` files (`DATA.zip`) from the MIT1003 page.

**1. ScanDiff data release.** Download it (about 53 GB) and rename the folders:

```bash
wget -O data.zip https://ailb-web.ing.unimore.it/publicfiles/ScanDiff_ICCV2025/data.zip
unzip data.zip -d scandiff_data

mkdir -p data
mv scandiff_data/task_embeddings.npy data/
mv scandiff_data/coco_freeview data/cocofreeview_edf
mv scandiff_data/mit1003 data/mit1003_edf
mv scandiff_data/cocosearch18 data/cocosearch18_edf

# COCO-Search18: one image folder per category (named with spaces) for both conditions,
# the target-absent training scanpaths, and the cluster files under their new names.
cd data/cocosearch18_edf
for d in images_tp/* images_ta/*; do
  c=$(basename "$d"); mkdir -p "images/${c//_/ }"; mv "$d"/* "images/${c//_/ }/"
done
rm -r images_tp images_ta
wget http://vision.cs.stonybrook.edu/~cvlab_download/coco_search18_fixations_TA_trainval.json
mv clusters_cocosearch18_tp_512_384.npy clusters_cocosearch18_edf_tp_512_384.npy
mv clusters_cocosearch18_ta_512_384.npy clusters_cocosearch18_edf_ta_512_384.npy
cd ../..
```

**2. COCO-Search18 and COCO-FreeView recordings.** Convert every EDF file to ASCII (`.asc`) with SR Research's tools. You can use `edf2asc -input` from the [EyeLink Developers Kit](https://www.sr-research.com/support/thread-13.html), or the EDF-to-ASC export of [EyeLink Data Viewer](https://www.sr-research.com/data-viewer/) with the `-input` option enabled. The `-input` option is required because trials are split on the `INPUT` events. Keep the original file names, which carry the subject number (for example `s6_cup.asc`, `s104_d.asc`). Put the EDF files, in any subfolder structure, inside `data/cocosearch18_edf/edfs/` and `data/cocofreeview_edf/edfs/`, then convert them in place:

```bash
find data/cocosearch18_edf/edfs data/cocofreeview_edf/edfs -name "*.edf" -exec edf2asc -input {} \;
```

**3. MIT1003 recordings.**

```bash
wget https://people.csail.mit.edu/tjudd/WherePeopleLook/DATA.zip
unzip DATA.zip -d data/mit1003_edf
```

The final layout is:

```text
data/
├── task_embeddings.npy
├── cocosearch18_edf/
│   ├── coco_search18_fixations_TP_{train,validation,test}.json
│   ├── coco_search18_fixations_TA_{trainval,test}.json
│   ├── clusters_cocosearch18_edf_{tp,ta}_512_384.npy
│   ├── images/<category>/
│   ├── dinov2_base_timm_image_features_{tp,ta}/<category>/
│   ├── SemSS/
│   └── edfs/**/*.asc
├── cocofreeview_edf/
│   ├── COCOFreeView_fixations_{train,validation,test}.json
│   ├── clusters_coco_freeview_512_384.npy
│   ├── images/
│   ├── dinov2_base_timm_image_features/
│   └── edfs/**/*.asc
└── mit1003_edf/
    ├── mit1003_fixations_{train,validation,test}.json
    ├── clusters_mit1003_512_384.npy
    ├── images/
    ├── dinov2_base_timm_image_features/
    └── DATA/<subject>/*.mat
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
