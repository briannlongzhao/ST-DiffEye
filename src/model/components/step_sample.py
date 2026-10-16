import numpy as np
import torch as th


class LossSecondMomentResampler:
    """Importance-samples diffusion timesteps proportionally to the RMS of their recent losses.

    Adapted from IDDPM (https://github.com/openai/improved-diffusion). Sampling is uniform
    until every timestep has ``history_per_term`` recorded losses.
    """

    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.num_timesteps = diffusion.num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([self.num_timesteps, history_per_term], dtype=np.float64)
        self._loss_counts = np.zeros([self.num_timesteps], dtype=np.int32)

    def weights(self):
        if not (self._loss_counts == self.history_per_term).all():
            return np.ones([self.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def sample(self, batch_size, device):
        """Return sampled timesteps and the importance weights that keep the loss unbiased."""
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights = th.from_numpy(1 / (len(p) * p[indices_np])).float().to(device)
        return indices, weights

    def update_with_local_losses(self, ts, losses):
        for t, loss in zip(ts.tolist(), losses.tolist()):
            if self._loss_counts[t] == self.history_per_term:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1
