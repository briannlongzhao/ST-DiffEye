import torch as th

from src.model.components.gaussian_diffusion import ModelMeanType, mean_flat
from src.model.components.respace import SpacedDiffusion


class CustomSpacedDiffusion(SpacedDiffusion):
    """Latent diffusion over gaze tokens with reconstruction, fixation-index and token-validity losses."""

    def __init__(self, lambda_x=1.0, lambda_y=1.0, lambda_t=1.0, lambda_f=1.0, use_timesteps=None, **kwargs):
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.lambda_t = lambda_t
        self.lambda_f = lambda_f
        super().__init__(use_timesteps, **kwargs)

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        model = self._wrap_model(model)
        model_kwargs = {} if model_kwargs is None else model_kwargs

        padding_mask = model_kwargs.pop("padding_mask").to(x_start.device)
        fixation_indices = model_kwargs["fixation_indices"]

        # Reconstruction target: (x, y, dt) plus the fixation index channel for data_channel.
        gt = x_start
        if model.model.fixation_strategy == "data_channel":
            gt = model.model.build_data_channel_scanpath(x_start, fixation_indices)

        latent_start = model.model.get_embeds(x_start, fixation_indices)
        if noise is None:
            noise = th.randn_like(latent_start)
        x_t = self.q_sample(latent_start, t, noise=noise)
        model_output = model(x_t, t, **model_kwargs)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=latent_start, x_t=x_t, t=t)[0],
            ModelMeanType.START_X: latent_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == latent_start.shape

        terms = {"mse": mean_flat((target - model_output) ** 2)}
        # Prior matching: the latent at the final timestep should have zero mean.
        out_mean, _, _ = self.q_mean_variance(
            latent_start, th.LongTensor([self.num_timesteps - 1]).to(latent_start.device)
        )
        tT_loss = mean_flat(out_mean**2)

        terms["reg_loss"], terms["t_loss"], terms["fixation_loss"] = self.reconstruction_loss(
            model.model.get_coords_and_time(latent_start), padding_mask, gt
        )
        terms["token_loss"] = self.token_validity_loss(
            latent_start, model.model.token_validity_predictor, padding_mask
        )
        terms["loss"] = (
            terms["mse"] + tT_loss + terms["reg_loss"] + terms["t_loss"] + terms["fixation_loss"] + terms["token_loss"]
        )
        return terms

    def reconstruction_loss(self, prediction, padding_mask, ground_truth):
        mask = padding_mask.unsqueeze(-1)
        pred = prediction * mask
        gt = ground_truth * mask
        valid_counts = padding_mask.sum(dim=1).clamp_min(1)

        x_loss = th.abs(pred[:, :, 0] - gt[:, :, 0]).sum(dim=1) / valid_counts
        y_loss = th.abs(pred[:, :, 1] - gt[:, :, 1]).sum(dim=1) / valid_counts
        reg_loss = x_loss * self.lambda_x + y_loss * self.lambda_y
        t_loss = th.abs(pred[:, :, 2] - gt[:, :, 2]).sum(dim=1) / valid_counts * self.lambda_t

        fixation_loss = th.zeros_like(t_loss)
        if pred.shape[2] > 3:  # data_channel also reconstructs the fixation index
            fixation_loss = th.abs(pred[:, :, 3:] - gt[:, :, 3:]).sum(dim=1).mean(dim=1) / valid_counts
            fixation_loss = fixation_loss * self.lambda_f
        return reg_loss, t_loss, fixation_loss
