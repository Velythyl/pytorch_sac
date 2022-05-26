import hydra
import torch
import tqdm
from torch import nn

from agent.gait import Gait
from agent.residual_actor import _ResidualActor
from utils import soft_update_params, mu_logstd_from_vector


class _Primitive(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

        self.device = None

    def init(self):
        return

    def forward(self, obs, frame_nb):
        raise NotImplementedError()

    def update(self, actor, batch):
        return

    def to(self, device):
        self.device = device
        return super(_Primitive, self).to(device)


class CompoundPrimitive(_Primitive):
    def __init__(self, primitives):
        super().__init__(primitives[0].action_dim)
        self.primitives = nn.ModuleList(primitives)

    def forward(self, obs, frame_nb):
        act = 0
        for pri in self.primitives:
            act = act + pri(obs, frame_nb)
        return act

    def to(self, device):
        for pri in self.primitives:
            pri.to(device)
        return super(CompoundPrimitive, self).to(device)


class _NumericPrimitive(_Primitive):
    def _forward(self, shape):
        raise NotImplementedError()

    def forward(self, obs, frame_nb):
        shape = obs.shape[0]

        ret = self._forward(shape)
        return ret


class NoopPrimitive(_NumericPrimitive):
    def _forward(self, shape):
        return torch.zeros(shape, self.action_dim, device=self.device)


class UniformPrimitive(_NumericPrimitive):
    def __init__(self, action_dim, bound):
        super().__init__(action_dim)
        self.bound = bound

    def _forward(self, shape):
        return torch.zeros(shape, self.action_dim, device=self.device).uniform_(-self.bound, self.bound)


class _NNBasedPrimitive(_Primitive):
    def __init__(self, action_dim, tau, nn):
        super().__init__(action_dim)
        self.tau = tau
        self.nn = nn

    def update(self, actor, batch):
        raise NotImplementedError()


class TargetPrimitive(_NNBasedPrimitive):
    def __init__(self, action_dim, tau, residual_target):
        super().__init__(action_dim, tau, nn=residual_target)

    def update(self, actor, batch):
        soft_update_params(actor.residual, self.nn, self.tau)

    def forward(self, obs, frame_nb):
        return mu_logstd_from_vector(self.nn(obs))[0]


class GaitPrimitive(_NNBasedPrimitive):
    def __init__(self, action_dim, tau, gait1, gait2):
        super(GaitPrimitive, self).__init__(action_dim, tau, nn=gait1)

        # Logic: nn is used by the actor. Always static, never backprops
        # Gait2 is an exact copy of nn. We update Gait2 using the actor (which uses nn).
        # That way, Gait2 can be backpropped "against" nn

        self.nn.requires_grad = False
        self.gait2 = gait2
        self.gait2.load_state_dict(self.nn.state_dict())
        self.got_init = False

        # in tuple so the nn.Module doesn't track them
        self.opt_loss = (
        torch.optim.Adam(self.gait2.parameters()), nn.MSELoss())

    def forward(self, obs, frame_nb):
        return self.nn(frame_nb)

    def update(self, actor, batch):
        obs = batch['obs']
        # next_obs = batch['next_obs']

        timestep = batch['timestep']
        # next_timestep = timestep + 1

        with torch.no_grad():
            y_target = actor(obs, timestep).mean  # torch.cat((obs, next_obs), dim=1)
        x = timestep  # torch.cat((timestep, next_timestep), dim=1)

        x.requires_grad = False
        y_target.requires_grad = False

        if not self.got_init:
            for i in tqdm.trange(10000):
                self.update_step(x, y_target)
            self.got_init = True

        self.update_step(x, y_target)
        self.nn.load_state_dict(self.gait2.state_dict())

    def update_step(self, x, y_target):
        opt, mse = self.opt_loss

        y_pred = self.gait2(x)

        opt.zero_grad()
        loss = mse(y_pred, y_target)
        loss.backward()
        opt.step()


class NoBackpropWrapper:
    def __init__(self, primitive):
        self.primitive = primitive

    def __call__(self, *args, **kwargs):
        return self.primitive(*args, **kwargs)

    def to(self, device):
        self.primitive.to(device)
        return self

    def update(self, actor, batch):
        return self.primitive.update(actor, batch)


def InstantiatePrimitives(action_dim, tau, which, gait_cfg):
    def _instantiate(residual_target):
        primitives = []
        if which.uniform:
            primitives += [UniformPrimitive(action_dim, which.uniform)]

        if which.target:
            primitives += [TargetPrimitive(action_dim, tau, residual_target)]

        if which.noop:
            primitives += [NoopPrimitive(action_dim)]

        if which.gait:
            gait1 = hydra.utils.instantiate(gait_cfg)
            gait2 = hydra.utils.instantiate(gait_cfg)

            primitives += [GaitPrimitive(action_dim, tau, gait1, gait2)]

        if len(primitives) > 1:
            primitive = CompoundPrimitive(primitives)
        elif len(primitives) == 1:
            primitive = primitives[0]
        else:
            primitive = None

        return primitive

    return _instantiate
