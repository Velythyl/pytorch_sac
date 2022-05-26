import numpy as np
import torch
from torch import nn as nn


TAU = np.pi * 2

class Gait(nn.Module):
    def __init__(self, nb_gaussians, action_len, n_frame_repeat, pow_2):
        super().__init__()
        self.nb_actuators = action_len
        self.mixture_dim = (action_len, nb_gaussians)

        self.pow_2 = pow_2
        if pow_2:
            n_frame_repeat = n_frame_repeat / 2

        # initial period is to have a ~25 frame period. Learnable parameter.
        self.period_b = nn.Parameter(torch.tensor(TAU / n_frame_repeat), requires_grad=False)

        # same init as https://github.com/studywolf/pydmps/blob/master/pydmps/dmp_rhythmic.py
        mu_matrix_init = np.linspace(0, TAU, nb_gaussians + 1)
        mu_matrix_init = mu_matrix_init[0:-1]
        mu_matrix_init = torch.tensor(action_len * [mu_matrix_init])

        self.mu_matrix = nn.Parameter(mu_matrix_init.float(), requires_grad=True)
        self.sigma_matrix = nn.Parameter(-torch.ones(self.mixture_dim), requires_grad=True)
        self.weights = nn.Parameter(torch.zeros(self.mixture_dim).uniform_(-self.nb_actuators, self.nb_actuators), requires_grad=True)



    def period(self):
        return TAU / self.period_b.clone().detach().item() / (2 if self.pow_2 else 1)

    def frame2percent(self, frame_nb):
        percent = frame_nb * self.period_b
        percent = torch.remainder(percent, TAU)
        percent = percent / TAU

        return percent

    def cyclic_gaussian_mixture(self, frame_numbers):
        # https://studywolf.wordpress.com/tag/rhythmic-dynamic-movement-primitives/
        x_mu = frame_numbers - self.mu_matrix
        cos_x_mu = torch.cos(self.period_b * x_mu)

        if self.pow_2:
            cos_x_mu = cos_x_mu ** 2

        scaled_x_mu_pow = torch.mul(self.sigma_matrix, cos_x_mu) - 1
        gaussian = torch.exp(scaled_x_mu_pow)
        return gaussian

    def forward(self, frame_nb):
        assert len(frame_nb.shape) == 2, "Gait always expects batch as input"

        frame_nb_batch = frame_nb.unsqueeze(-1).expand(frame_nb.shape[0], *self.mixture_dim)

        mixed = self.cyclic_gaussian_mixture(frame_nb_batch)
        mixed_weighted = torch.mul(mixed, self.weights)

        mixed = torch.sum(mixed, dim=-1)
        mixed_weighted = torch.sum(mixed_weighted, dim=-1)

        activations = mixed_weighted / mixed

        return activations

class NNGait(nn.Module):
    def __init__(self, hidden_dim, action_shape, n_frame_repeat):
        super().__init__()
        self.nb_actuators = action_shape[0]
        self.gaussian = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_shape[0]),
            #nn.Tanh()
            #nn.Sigmoid()
        )
        self.period_b = nn.Parameter(torch.tensor(TAU / n_frame_repeat), requires_grad=True)

    def frame2percent(self, frame_nb):
        percent = frame_nb * self.period_b
        percent = torch.remainder(percent, TAU)
        percent = percent / TAU

        return percent

    def period(self):
        return TAU / self.period_b.clone().detach().item()

    def forward(self, frame_nb):
        frame_nb = self.frame2percent(frame_nb)
        assert len(frame_nb.shape) == 2, "Gait always expects batch as input"
        return self.gaussian(frame_nb)

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    FRAMES = 50
    gait = Gait(500, [2], 50).cuda()#Gait(300, [12], n_frame_repeat=FRAMES).cuda()
    opt = torch.optim.Adam(gait.parameters(), lr=0.1)

    x = torch.arange(0, FRAMES, requires_grad=False).unsqueeze(-1).cuda()

    y = torch.normal(0.2, 1, (int(FRAMES / 2), 2), requires_grad=False).cuda()
    y2 = torch.normal(0.2, 1, (int(FRAMES / 2), 2), requires_grad=False).cuda()
    y = y + y2
    y = torch.cat((y, y))

    target = y

    plt.plot(x.cpu().numpy(), y.cpu().numpy())
    # plt.plot(x, torch.cos(x))
    plt.plot(gait(x).detach().cpu().numpy())
    plt.show()

    before = gait.period()

    mse = torch.nn.MSELoss()
    for i in range(100):
        y_z_pred = gait(x)

        if i % 10 == 0:
            plt.plot(x.cpu().numpy(), y_z_pred.clone().detach().cpu().numpy())
            # plt.plot(x, torch.cos(x))
            plt.plot(x.detach().cpu().numpy(), y_z_pred.detach().cpu().numpy())
            plt.show()
#        DEBUG = y_z_pred.squeeze().detach().numpy()

        loss = mse(y_z_pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.detach().item())

    print('before', before)
    print('after', gait.period())

    plt.plot(x.cpu().numpy(), y.cpu().numpy())
    # plt.plot(x, torch.cos(x))
    x = torch.linspace(0, FRAMES * 2, 5000).unsqueeze(-1).cuda()
    y_z_pred = gait(x)
    plt.plot(x.detach().cpu().numpy(), y_z_pred.detach().cpu().numpy())
    plt.show()




    plot_gait(gait, 10000, as_tb=False, save_dir='./temp')

    pass
