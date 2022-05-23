import torch
from torch import nn

from agent.residual_actor import _ResidualActor
from utils import soft_update_params


class _Primitive(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        
        self.device = None

    def forward(self, obs, frame_nb):
        raise NotImplementedError()

    def update(self, batch):
        return
    


class CompoundPrimitive(_Primitive):
    def __init__(self, primitives):
        super().__init__(primitives[0].action_dim)
        self.primitives = primitives

    def forward(self, obs, frame_nb):
        act = self.primitives[0](obs, frame_nb)
        for pri in self.primitives[1:]:
            act = act + pri(obs, frame_nb)
        return act


class _NumericPrimitive(_Primitive):
    def _forward(self, shape):
        raise NotImplementedError()

    def forward(self, obs, frame_nb):
        shape = obs.shape[0]

        ret = self._forward(shape)
        return ret

    def to(self, device):
        self.device = device
        return super(_Primitive, self).to(device)

class NoopPrimitive(_NumericPrimitive):
    def _forward(self, shape):
        return torch.zeros(shape, self.action_dim, device=self.device)


class UniformPrimitive(_NumericPrimitive):
    def __init__(self, action_dim, bound):
        super().__init__(action_dim)
        self.bound = bound

    def _forward(self, shape):
        return torch.zeros(shape, self.action_dim, device=self.device).uniform_(-self.bound, self.bound)


class _ActorBasedPrimitive(_Primitive):
    def __init__(self, action_dim, tau):
        super().__init__(action_dim)
        self.tau = tau

    def update(self, batch):
        raise NotImplementedError()


class TargetPrimitive(_ActorBasedPrimitive):
    def __init__(self, action_dim, tau, actor_target):
        super().__init__(action_dim, tau)
        self.actor_target = actor_target

    def update(self, batch, actor):
        soft_update_params(actor, self.actor_target, self.tau)

    def forward(self, obs, frame_nb):
        return self.actor_target(obs).mean


def InstantiatePrimitives(action_dim, tau, which):

    def _instantiate(residual_target):
        primitives = []
        if which.uniform:
            primitives += [UniformPrimitive(action_dim, which.uniform)]

        if which.target:
            primitives += [TargetPrimitive(action_dim, tau, residual_target)]

        if which.noop:
            primitives += [NoopPrimitive(action_dim)]

        if len(primitives) > 1:
            return CompoundPrimitive(primitives)
        elif len(primitives) == 1:
            return primitives[0]
        else:
            return None

    return _instantiate