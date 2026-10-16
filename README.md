# ST-DiffEye

Official code for **ST-DiffEye: Diffusion-based Continuous Gaze Generation via Joint Scanpath-Trajectory Modeling**.

[Paper](https://arxiv.org/abs/2606.15486) · [Project page](https://st-diffeye.github.io/) · [Checkpoints](https://huggingface.co/briannlongzhao/ST-DiffEye)

ST-DiffEye is a diffusion model that generates continuous eye-tracking trajectories, conditioned on an image and an optional visual-search target. During training, each trajectory point is augmented with a continuous *fixation index* that aligns it with the human scanpath. At inference, the generated trajectories are converted to scanpaths with a fixation-detection algorithm and evaluated with CRPS-based and standard scanpath metrics.

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

| Checkpoint | Benchmark | Experiment preset | Trained on | Fixation-index concatenation | Trajectory length |
|---|---|---|---|---|---|
| `COCOSearch18-TP.pt` | COCO-Search18, target present | `cocosearch18_tp` | COCO-Search18 TP + TA | raw data channel | 750 |
| `COCOSearch18-TA.pt` | COCO-Search18, target absent | `cocosearch18_ta` | COCO-Search18 TA | raw data channel | 750 |
| `COCOFreeView.pt` | COCO-FreeView | `cocofreeview` | COCO-FreeView + MIT1003 | raw data channel | 1500 |
| `MIT1003.pt` | MIT1003 | `mit1003` | COCO-FreeView + MIT1003 | **latent space** | 750 |

Each experiment preset (`configs/experiment/`) holds all the settings of its checkpoint, so commands only need the preset name.

## Data

Data download instructions will be added soon. All scripts read the datasets from one folder passed as `data_dir=/path/to/data`, with this layout:

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

Trained jointly on the target-present and target-absent training sets. The fixation index is concatenated to the trajectory as a raw input channel.

```bash
# Train
python src/train.py experiment=cocosearch18_tp data_dir=/path/to/data
# Evaluate the released checkpoint
python src/eval.py experiment=cocosearch18_tp data_dir=/path/to/data ckpt_path=checkpoints/COCOSearch18-TP.pt
```

### COCO-Search18, target absent

Trained on the target-absent training set for 500 epochs. The fixation index is concatenated to the trajectory as a raw input channel.

```bash
# Train
python src/train.py experiment=cocosearch18_ta data_dir=/path/to/data
# Evaluate the released checkpoint
python src/eval.py experiment=cocosearch18_ta data_dir=/path/to/data ckpt_path=checkpoints/COCOSearch18-TA.pt
```

### COCO-FreeView

Trained jointly on COCO-FreeView and MIT1003 with 1500-point trajectories (batch size 64). The fixation index is concatenated to the trajectory as a raw input channel.

```bash
# Train
python src/train.py experiment=cocofreeview data_dir=/path/to/data
# Evaluate the released checkpoint
python src/eval.py experiment=cocofreeview data_dir=/path/to/data ckpt_path=checkpoints/COCOFreeView.pt
```

### MIT1003

Trained jointly on COCO-FreeView and MIT1003.

> **Note:** unlike the other three datasets, MIT1003 uses **latent-space concatenation** of the fixation index (`model.fixation_strategy=latent_space`, set by the `mit1003` preset). The trajectory is first encoded into a 256-dimensional latent, and the fixation index is appended as a 257th latent channel instead of a 4th raw input channel. This works better on MIT1003, whose raw trajectories are noisier (see Appendix C.2 of the paper).

```bash
# Train
python src/train.py experiment=mit1003 data_dir=/path/to/data
# Evaluate the released checkpoint
python src/eval.py experiment=mit1003 data_dir=/path/to/data ckpt_path=checkpoints/MIT1003.pt
```

### Outputs

Training writes checkpoints every 50 epochs (`epoch_<E>.pt`) and after every epoch (`last.pt`) to `logs/train/<preset>_<date>_<time>/checkpoints/`. The released checkpoints are COCOSearch18-TP at epoch 250, COCOSearch18-TA at epoch 450, COCOFreeView at epoch 200 and MIT1003 at epoch 200. To evaluate your own model, use the same preset with `ckpt_path=<path to checkpoint>`.

Evaluation writes to `logs/eval/<preset>_<date>_<time>/`:

- `generations_epoch_<E>/generations_<Dataset>_test.pkl`: for every test stimulus, one generated sample per human viewer, as normalized `trajectories` (x, y, dt) and the extracted `scanpaths` (x, y, duration).
- `generations_epoch_<E>/original_<Dataset>_test.pkl`: the human scanpaths.
- `metrics_epoch_<E>/metrics_<Dataset>_test.json`: CRPS, Mean, Best and KLD values for every metric.

### Options

Append any of these to the commands above:

| Option | Effect |
|---|---|
| `data.max_samples=100` | Use only the first 100 samples of each dataset (quick check). |
| `seed=<int>` | Sampling / training seed (default `0`). |
| `evaluation.data_to_extract=[preds]` | Only generate, without computing metrics. |
| `evaluation.data_to_extract=[preds,metrics,qualitatives]` | Also draw the generated scanpaths on the images. |
| `hydra.run.dir=<existing eval folder>` | Reuse the generations saved there and only recompute the metrics. |
| `ckpt_path=<path to last.pt>` (training) | Resume training. |
| `logger=wandb` (training) | Log losses to Weights & Biases. |
| `data.batch_size_train=<int>` (training) | Change the batch size. The released models were trained on a single NVIDIA GH200 GPU; `cocofreeview` (1500-point trajectories, batch size 64) needs the most memory. |

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
