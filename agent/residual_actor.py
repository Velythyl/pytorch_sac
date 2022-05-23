import torch
from torch import nn

import utils
from agent.actor import SquashedNormal


class _ResidualActor(nn.Module):
    def __init__(self, residual, primitive, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()
        self.residual = residual
        self.log_std_bounts = log_std_bounds
        self.log_std_bounts = residual.log_std_bounds
        self.primitive = primitive

    def _forward(self, obs, frame_nb) -> torch.Tensor:
        return NotImplementedError()

    def forward(self, obs, frame_nb):
        mu, log_std = self._forward(obs, frame_nb).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def _log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_residual/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_residual/fc{i}', m, step)

    def log(self, logger, step):
        self._log(logger, step)
        self.residual.log(logger, step)


class NoResidualActor(_ResidualActor):
    def _forward(self, obs, frame_nb):
        return self.residual(obs)

    def log(self, logger, step):
        self._log(logger, step)


class ResidualActor(_ResidualActor):
    def __init__(self, residual, primitive, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__(residual, primitive, action_dim, hidden_dim, hidden_depth, log_std_bounds)

        obs_dim = action_dim * 3

        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)
        self.trunk.apply(utils.identity_init)

    def _forward(self, obs, frame_nb):
        residual = self.residual(obs)
        with torch.no_grad():
            base = self.primitive(obs, frame_nb).detach()

        obs = torch.cat((residual, base), dim=-1)
        mixed = self.trunk(obs)
        return mixed