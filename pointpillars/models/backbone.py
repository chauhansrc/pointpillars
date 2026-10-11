import torch
import torch.nn as nn


def _bn(c):
    return nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)


def conv_bn_relu(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), _bn(cout), nn.ReLU(inplace=True))


class Backbone(nn.Module):
    """Top-down blocks Block(S, L, F) + upsampling Up(S_in, S_out, F), concatenated (paper Sec. 2.2).

    Defaults are the paper's car setting: three blocks at strides 2/4/8 relative to the pillar grid
    with 4/6/6 convs, each upsampled back to stride 2 with 128 channels -> 384-channel output.
    """

    def __init__(self, in_channels=64,
                 blocks=((64, 4, 2), (128, 6, 2), (256, 6, 2)),     # (channels, layers, stride)
                 up_channels=(128, 128, 128),
                 up_strides=(1, 2, 4)):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        cin = in_channels
        for (cout, layers, stride), uc, us in zip(blocks, up_channels, up_strides):
            self.blocks.append(nn.Sequential(conv_bn_relu(cin, cout, 3, stride, 1),
                                             *[conv_bn_relu(cout, cout) for _ in range(layers - 1)]))
            self.ups.append(nn.Sequential(nn.ConvTranspose2d(cout, uc, us, us, bias=False),
                                          _bn(uc), nn.ReLU(inplace=True)))
            cin = cout
        self.out_channels = sum(up_channels)

    def forward(self, x):
        outs = []
        for block, up in zip(self.blocks, self.ups):
            x = block(x)
            outs.append(up(x))
        return torch.cat(outs, 1)
