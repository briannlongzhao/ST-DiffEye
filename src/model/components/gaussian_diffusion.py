# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import enum
import math

import numpy as np
import torch as th
import torch.nn as nn


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):
    """Which type of output the model predicts."""

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """Get a pre-defined beta schedule for the given name."""
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        return np.linspace(scale * 0.0001, scale * 0.02, num_diffusion_timesteps, dtype=np.float64)
    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
    if schedule_name == "sqrt":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: 1 - np.sqrt(t + 0.0001))
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Discretize alpha_bar(t), the cumulative product of (1 - beta) over t in [0, 1], into betas."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """Gaussian diffusion with fixed (large) variances and ancestral sampling.

    :param betas: a 1-D numpy array of betas for each diffusion timestep.
    :param model_mean_type: what the model predicts (see ModelMeanType).
    """

    def __init__(self, *, betas, model_mean_type):
        self.model_mean_type = model_mean_type

        betas = np.array(betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # q(x_t | x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

        # Fixed-large model variance: beta_t, with the first entry set to the posterior variance.
        self.model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        self.model_log_variance = np.log(self.model_variance)

    def q_mean_variance(self, x_start, t):
        """Mean, variance and log-variance of q(x_t | x_0)."""
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)."""
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_variance

    def p_mean_variance(self, model, x, t, clip_denoised=False, model_kwargs=None):
        """Apply the model to get p(x_{t-1} | x_t) and a prediction of x_0."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        assert t.shape == (x.shape[0],)
        model_output = model(x, t, **model_kwargs)

        model_log_variance = _extract_into_tensor(self.model_log_variance, t, x.shape)
        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        else:
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        model_mean, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return {"mean": model_mean, "log_variance": model_log_variance, "pred_xstart": pred_xstart}

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_sample(self, model, x, t, clip_denoised=False, model_kwargs=None):
        """Sample x_{t-1} from the model at timestep t."""
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        noise = th.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=False, model_kwargs=None, device=None, progress=False):
        """Generate samples by running the full reverse process from noise."""
        if device is None:
            device = next(model.parameters()).device
        sample_x = noise if noise is not None else th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                sample_x = self.p_sample(
                    model, sample_x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
                )["sample"]
        return sample_x

    def token_validity_loss(self, sample_x, token_predictor, padding_mask):
        """Cross-entropy between predicted token validity and the ground-truth padding mask."""
        logits = token_predictor(sample_x)
        target = padding_mask.unsqueeze(-1).reshape(-1, 1).squeeze()
        loss = nn.CrossEntropyLoss(reduction="none")(logits.reshape(-1, 2), target)
        return loss.view(padding_mask.shape).mean(dim=-1)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract values from a 1-D numpy array for a batch of indices, broadcast to broadcast_shape."""
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)
