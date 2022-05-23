import torch
from torch import nn

from agent.residual_actor import _ResidualActor
from utils import soft_update_params


class _Primitive(nn.Module):
    def __init__(self, action_dim, tau):
        super().__init__()
        self.action_dim = action_dim
        self.tau = tau

    def forward(self, obs, frame_nb):
        raise NotImplementedError()

    def update(self, batch):
        return

class CompoundPrimitive(_Primitive):
    def __init__(self, primitives):
        super().__init__(None, None)
        self.primitives = primitives

    def forward(self, obs, frame_nb):
        act = self.primitives[0](obs, frame_nb)
        for pri in self.primitives[1:]:
            act = act + pri(obs, frame_nb)
        return act


class NoopPrimitive(_Primitive):
    def forward(self, obs, frame_nb):
        return torch.zeros(self.action_dim)


class UniformPrimitive(_Primitive):
    def __init__(self, action_dim, tau, bound):
        super().__init__(action_dim, tau)
        self.bound = bound

    def forward(self, obs, frame_nb):
        return torch.rand(-self.bound, self.bound, self.action_dim)


class _ActorBasedPrimitive(_Primitive):
    def __init__(self, actor_target: _ResidualActor, actor, action_dim, tau):
        super().__init__(action_dim, tau)
        self.actor_target = actor_target
        self.actor = actor

    def update(self, batch):
        raise NotImplementedError()


class TargetPrimitive(_ActorBasedPrimitive):
    def update(self, batch):
        soft_update_params(self.actor, self.actor_target, self.tau)

    def forward(self, obs, frame_nb):
        return self.actor_target(obs).mean


