import torch
from torch import nn

from agent.residual_actor import _ResidualActor
from utils import soft_update_params, mu_logstd_from_vector


class _Primitive(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

        self.device = None

    def forward(self, obs, frame_nb):
        raise NotImplementedError()

    def update(self, actor, batch):
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


class _ResidualBasedPrimitive(_Primitive):
    def __init__(self, action_dim, tau):
        super().__init__(action_dim)
        self.tau = tau

    def update(self, actor, batch):
        raise NotImplementedError()


class TargetPrimitive(_ResidualBasedPrimitive):
    def __init__(self, action_dim, tau, residual_target):
        super().__init__(action_dim, tau)
        self.residual_target = residual_target

    def update(self, actor,batch):
        soft_update_params(self.actor.residual, self.residual_target, self.tau)

    def forward(self, obs, frame_nb):
        return mu_logstd_from_vector(self.residual_target(obs))[0]


class NoBackpropWrapper:
    def __init__(self, primitive):
        self.primitive = primitive

    def __call__(self, obs, frame_nb):
        return self.primitive(obs, frame_nb)

    def update(self, actor, batch):
        return self.primitive.update(actor, batch)

    def to(self, device):
        self.primitive.to(device)
        return self


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
            primitive = CompoundPrimitive(primitives)
        elif len(primitives) == 1:
            primitive = primitives[0]
        else:
            primitive = None

        if primitive is not None:
            primitive = NoBackpropWrapper(primitive)

        return primitive

    return _instantiate
