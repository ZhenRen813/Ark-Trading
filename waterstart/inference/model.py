from math import log2
from statistics import fmean

import torch
import torch.nn as nn


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    return x + torch.detach_(x.clamp(min, max) - x)


class GatedTransition(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.softplus = nn.Softplus()

        self.lin_xr = nn.Linear(input_dim, hidden_dim)
        self.lin_hr = nn.Linear(z_dim, hidden_dim)

        self.lin_xm_ = nn.Linear(input_dim, z_dim)
        self.lin_rm_ = nn.Linear(hidden_dim, z_dim)

        self.lin_xg = nn.Linear(input_dim, z_dim)
        self.lin_hg = nn.Linear(z_dim, z_dim)

        self.lin_hm = nn.Linear(z_dim, z_dim)

        self.lin_m_s = nn.Linear(z_dim, z_dim)

    def forward(  # type: ignore
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r = torch.relu_(self.lin_xr(x) + self.lin_hr(h))
        mean_: torch.Tensor = self.lin_xm_(x) + self.lin_rm_(r)

        g = torch.sigmoid_(self.lin_xg(x) + self.lin_hg(h))

        mean: torch.Tensor = (1 - g) * self.lin_hm(h) + g * mean_
        sigma = self.softplus(self.lin_m_s(mean_.relu()))

        return mean, sigma


class Emitter(nn.Module):
    def __init__(self, z_dim: int, n_cur: int, hidden_dim: int):
        super().__init__()
        self.n_cur = n_cur
        self.lin1 = nn.Linear(z_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_logits = nn.Linear(hidden_dim, n_cur)
        self.lin_fraction = nn.Linear(hidden_dim, n_cur)

    def forward(  # type: ignore
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # z: (..., z_dim)

        out = self.lin1(z).relu_()
        out = self.lin2(out).relu_()

        return self.lin_logits(out), self.lin_fraction(out).tanh()


# TODO: should we keep this name?
class CNN(nn.Module):
    def __init__(
        self,
        batch_size: int,
        window_size: int,
        market_features: int,
        out_features: int,
        n_sym: int,
        max_trades: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.window_size = window_size
        self.market_features = market_features
        self.kernel_size = 3
        self.n_sym = n_sym
        self.max_trades = max_trades
        self.prev_step_features = 2 * n_sym * max_trades + 1

        hidden_dim = 2 ** max(
            5, round(log2(fmean((market_features, self.prev_step_features))))
        )

        self.conv1 = nn.Conv1d(
            market_features, market_features, kernel_size=self.kernel_size
        )
        self.conv2 = nn.Conv1d(
            market_features,
            hidden_dim,
            kernel_size=window_size + 1 - self.kernel_size,
        )
        self.lin1 = nn.Linear(self.prev_step_features, hidden_dim)
        self.lin2 = nn.Linear(2 * hidden_dim, out_features)

    def forward(  # type: ignore
        self,
        market_data: torch.Tensor,
        prev_step_data: torch.Tensor,
    ) -> torch.Tensor:
        # market_data: (batch_size, market_features, window_size)
        # prev_step_data: (batch_size, prev_step_features)

        out1: torch.Tensor = self.conv1(market_data).relu_()
        out1 = self.conv2(out1).squeeze(2).relu_()

        out2: torch.Tensor = self.lin1(prev_step_data)

        out = torch.cat((out1, out2), dim=1)
        out = self.lin2(out)

        return out


# TODO: this won't stay here..
# TODO: maybe use 3 layers?
class NeuralBaseline(nn.Module):
    def __init__(self, n_features: int, z_dim: int, hidden_dim: int, n_cur: int):
        super().__init__()
        self.lin1 = nn.Linear(n_features + z_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, n_cur)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # x: (..., 2 * n_cur * (max_trades + 1) + z_dim + 1)

        out = self.lin1(x).relu_()
        out = self.lin2(out)

        return out
