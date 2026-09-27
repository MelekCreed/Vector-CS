"""Small temporal classifiers (optional; requires PyTorch).

Deliberately tiny: a few thousand parameters, CPU inference well under a
millisecond per sequence. Two architectures to compare:
  * GRU   — recurrent, good at order/timing
  * TCN   — dilated 1-D convolutions, parallel and stable to train
"""

from __future__ import annotations

import numpy as np


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("learned models need PyTorch: pip install -e .[learned]") from exc


def build(kind: str, n_features: int, n_classes: int, hidden: int = 48):
    torch = _torch()
    nn = torch.nn

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(n_features)
            self.gru = nn.GRU(n_features, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, n_classes))

        def forward(self, x):
            _, h = self.gru(self.norm(x))
            return self.head(h[-1])

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(n_features)
            layers, c = [], n_features
            for d in (1, 2, 4):
                layers += [nn.Conv1d(c, hidden, 3, padding=d, dilation=d), nn.GELU(), nn.Dropout(0.1)]
                c = hidden
            self.net = nn.Sequential(*layers)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x):
            h = self.net(self.norm(x).transpose(1, 2))
            return self.head(h.mean(dim=2))

    return {"gru": GRUNet, "tcn": TCN}[kind]()


def train(kind: str, X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int = 60,
          lr: float = 3e-3, seed: int = 0, augment: bool = True):
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(kind, X.shape[2], n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.long)
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(len(X))
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            xb = Xt[idx]
            if augment:   # small noise + time-warp-ish jitter against overfitting
                xb = xb + 0.02 * torch.randn_like(xb)
                shift = int(rng.integers(-2, 3))
                xb = torch.roll(xb, shift, dims=1)
            opt.zero_grad()
            loss = lossf(model(xb), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    torch = _torch()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X)), dim=1).numpy()
