import torch
from torch import nn

import utils
from agent.actor import SquashedNormal


class _ResidualActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()

        self.residual = utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)
        self.residual.apply(utils.weight_init)
        self.log_std_bounds = log_std_bounds
        self.outputs = dict()

    def residual_actor(self, obs):
        return self.residual(obs)

    def _forward(self, obs, frame_nb) -> (torch.Tensor, torch.Tensor):
        """
        returns mu, log_std
        """
        return NotImplementedError()

    def forward(self, obs, frame_nb):
        mu, log_std = self._forward(obs, frame_nb)

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

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_residual/{k}_hist', v, step)

        for i, m in enumerate(self.residual):
            if type(m) == nn.Linear:
                logger.log_param(f'train_residual/fc{i}', m, step)

class NoPrimitiveResidualActor(_ResidualActor):
    def _forward(self, obs, _):
        return utils.mu_logstd_from_vector(self.residual(obs))

class BarePrimitiveResidualActor(_ResidualActor):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds, primitive):
        super().__init__(obs_dim + action_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds)
        self.primitive = primitive

    def _forward(self, obs, frame_nb):
        with torch.no_grad():
            base = self.primitive(obs, frame_nb).detach()

        obs_and_primitive = torch.cat((obs, base), dim=1)
        residual = self.residual(obs_and_primitive)

        mu, log_std = utils.mu_logstd_from_vector(residual)
        mu = mu + base
        return mu, log_std

    def log(self, logger, step):
        self.primitive.actor_log(logger, step)
        super(BarePrimitiveResidualActor, self).log(logger, step)

class MixedPrimitiveResidualActor(_ResidualActor):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds, primitive):
        super().__init__(obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds)

        self.primitive = primitive

        obs_dim = action_dim * 3

        self.mixer = utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)
        self.mixer.apply(utils.identity_init)

    def _forward(self, obs, frame_nb):
        #frame_nb = NotImplementedError()

        residual = self.residual(obs)
        with torch.no_grad():
            base = self.primitive(obs, frame_nb).detach()

        obs = torch.cat((residual, base), dim=-1)
        mixed = self.mixer(obs)

        mu, log_std = utils.mu_logstd_from_vector(mixed)
        mu = mu + base
        return mu, log_std

    def log(self, logger, step):
        self.primitive.actor_log(logger, step)
        super(MixedPrimitiveResidualActor, self).log(logger, step)

def InstantiateResidualActor(action_dim, obs_dim, hidden_dim, hidden_depth, log_std_bounds, bare):
    def _instantiate(primitive):
        if primitive is None:
            residual_actor = NoPrimitiveResidualActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, hidden_depth=hidden_depth, log_std_bounds=log_std_bounds)

        elif bare:
            residual_actor = BarePrimitiveResidualActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, hidden_depth=hidden_depth, log_std_bounds=log_std_bounds, primitive=primitive)
        else:
            residual_actor = MixedPrimitiveResidualActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, hidden_depth=hidden_depth, log_std_bounds=log_std_bounds, primitive=primitive)
        return residual_actor

    return _instantiate