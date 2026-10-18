"""Generate gaze trajectories and scanpaths for a single image with a released checkpoint.

Saves one figure with the generated trajectories (left) and scanpaths (right) overlaid on the image.

Examples:
    # Visual search for a target (one of the 18 COCO-Search18 categories)
    python src/demo.py --image path/to/image.jpg --task cocosearch18_tp --target clock
    # Free viewing
    python src/demo.py --image path/to/image.jpg --task cocofreeview
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import timm
import torch
import hydra
from hydra import compose, initialize_config_dir
from PIL import Image

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.callbacks.model_checkpoint import strip_legacy_keys
from src.data.components.collator import PAD_VALUE
from src.evaluator import convert_trajectory_to_scanpath
from src.utils.create_diffusion import create_diffusion
from src.utils.train_utils import get_device, seed_everything

# Task setting -> released checkpoint and whether a search target is required.
TASKS = {
    "cocosearch18_tp": dict(ckpt="COCOSearch18-TP.pt", search=True),
    "cocosearch18_ta": dict(ckpt="COCOSearch18-TA.pt", search=True),
    "cocofreeview": dict(ckpt="COCOFreeView.pt", search=False),
    "mit1003": dict(ckpt="MIT1003.pt", search=False),
}
SEARCH_TARGETS = [
    "bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
    "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign", "toilet", "tv",
]
# COCO stimuli were shown full screen at 1680x1050; MIT1003 stimuli at their own size.
COCO_FRAME = (1680.0, 1050.0)


def extract_image_features(image, device):
    """DINOv2 (ViT-B/14, 4 registers) patch features, as used for the ScanDiff data release."""
    model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0).eval().to(device)
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
    with torch.no_grad():
        features = model.forward_features(transform(image.resize((518, 518))).unsqueeze(0).to(device))
    return features[:, model.num_prefix_tokens:]  # drop CLS and register tokens -> (1, 1369, 768)


def load_model(task, ckpt_path, device):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="eval.yaml", overrides=[f"experiment={task}"])
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(strip_legacy_keys(ckpt["model"]))
    return model.to(device).eval(), create_diffusion(cfg.diffusion)


def generate(model, diffusion, img_feats, task_embedding, num_samples, device):
    """Sample trajectories and convert them to scanpaths, following ``Evaluator.extract_predictions``."""
    model_kwargs = dict(
        y=img_feats.repeat(num_samples, 1, 1),
        task_embedding=task_embedding.to(device).repeat(num_samples, 1),
    )
    noise = torch.randn(num_samples, model.max_len, model.diffusion_latent_dim, device=device)
    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            model, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        trajectories = model.get_coords_and_time(samples)[:, :, :3]
        # Pad every point after the first one predicted as invalid.
        valid = (model.token_validity_predictor(samples).softmax(dim=-1)[..., 1] >= 0.5).long()
        lengths = torch.cumprod(valid, dim=-1).sum(-1)
        trajectories[torch.arange(model.max_len, device=device)[None] >= lengths[:, None]] = PAD_VALUE
    return trajectories


def unpad(seq):
    seq = seq.cpu().numpy()
    return seq[seq[:, 0] != PAD_VALUE]


def save_figure(image, trajectories, scanpaths, title, out_path):
    width, height = image.size
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[np.arange(len(trajectories)) % 10]
    fig, (ax_traj, ax_scan) = plt.subplots(1, 2, figsize=(16, 16 * height / (2 * width) + 0.8))
    for ax, name in [(ax_traj, "Trajectories"), (ax_scan, "Scanpaths")]:
        ax.imshow(image)
        ax.set_title(name)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")

    for color, traj in zip(colors, trajectories):
        traj = unpad(traj)
        ax_traj.plot(traj[:, 0] * width, traj[:, 1] * height, "-", color=color, lw=1.2, alpha=0.85)

    for i, (color, scan) in enumerate(zip(colors, scanpaths)):
        scan = unpad(scan)
        x, y, dur = scan[:, 0] * width, scan[:, 1] * height, scan[:, 2]
        ax_scan.plot(x, y, "-", color=color, lw=1.2, alpha=0.85)
        # Marker area grows with fixation duration (seconds).
        ax_scan.scatter(x, y, s=40 + 600 * dur, color=color, alpha=0.6, edgecolors="white", linewidths=0.8,
                        label=f"sample {i + 1}")
    ax_scan.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="input image")
    parser.add_argument("--task", required=True, choices=list(TASKS), help="task setting (selects the checkpoint)")
    parser.add_argument("--target", choices=SEARCH_TARGETS, help="search target, required for cocosearch18_tp/ta")
    parser.add_argument("--ckpt", help="checkpoint (default: checkpoints/<released checkpoint for --task>)")
    parser.add_argument("--task_embeddings", default=str(ROOT / "data" / "task_embeddings.npy"))
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, help="random seed (default: none, different samples on every run)")
    parser.add_argument("--out", help="output figure (default: <image name>_<task>[_<target>].png)")
    args = parser.parse_args()

    task = TASKS[args.task]
    if task["search"] and args.target is None:
        parser.error(f"--target is required for --task {args.task}")
    if not task["search"] and args.target is not None:
        parser.error(f"--task {args.task} is free viewing and does not take --target")
    ckpt_path = args.ckpt or str(ROOT / "checkpoints" / task["ckpt"])
    out_path = args.out or "_".join([Path(args.image).stem, args.task] + ([args.target.replace(" ", "_")] if args.target else [])) + ".png"

    if args.seed is not None:
        seed_everything(args.seed)
    device = get_device()
    image = Image.open(args.image).convert("RGB")
    task_embedding = torch.from_numpy(np.load(args.task_embeddings, allow_pickle=True).item()[args.target or ""])

    img_feats = extract_image_features(image, device)
    model, diffusion = load_model(args.task, ckpt_path, device)
    trajectories = generate(model, diffusion, img_feats, task_embedding, args.num_samples, device)
    frame = image.size if args.task == "mit1003" else COCO_FRAME
    scanpaths = convert_trajectory_to_scanpath(trajectories, max_len=16, device=device, normalization_size=frame)

    title = f"{args.task}: search for '{args.target}'" if args.target else f"{args.task}: free viewing"
    save_figure(image, trajectories, scanpaths, title, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
