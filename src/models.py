"""Compact residual network for 12-lead, 100 Hz ECG classification."""
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(incoming, outgoing, 7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(outgoing), nn.ReLU(),
            nn.Conv1d(outgoing, outgoing, 7, padding=3, bias=False),
            nn.BatchNorm1d(outgoing),
        )
        self.skip = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(
            nn.Conv1d(incoming, outgoing, 1, stride=stride, bias=False),
            nn.BatchNorm1d(outgoing),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class ECGResNet(nn.Module):
    def __init__(self, classes=5, width=32, blocks_per_stage=2):
        super().__init__()
        layers = [nn.Conv1d(12, width, 15, stride=2, padding=7, bias=False),
                  nn.BatchNorm1d(width), nn.ReLU()]
        incoming = width
        for stage, outgoing in enumerate([width, width * 2, width * 4]):
            for block in range(blocks_per_stage):
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(ResidualBlock(incoming, outgoing, stride))
                incoming = outgoing
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(incoming, classes)

    def forward(self, x):
        return self.head(self.features(x).squeeze(-1))
