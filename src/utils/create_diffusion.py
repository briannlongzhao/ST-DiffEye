# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from src.model.components import gaussian_diffusion as gd
from src.model.components.diffusion_classes.custom_gaussian_diffusion import CustomSpacedDiffusion
from src.model.components.respace import space_timesteps


def create_diffusion(diffusion_cfg):
    """Build the diffusion process from the ``diffusion`` config (steps, noise schedule, prediction target)."""
    steps = diffusion_cfg.num_timesteps
    return CustomSpacedDiffusion(
        use_timesteps=space_timesteps(steps, [steps]),
        betas=gd.get_named_beta_schedule(diffusion_cfg.noise_schedule, steps),
        model_mean_type=gd.ModelMeanType.START_X if diffusion_cfg.predict_xstart else gd.ModelMeanType.EPSILON,
    )
