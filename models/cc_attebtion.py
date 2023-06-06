import torch
import torch.nn as nn
import torch.nn.functional as F



def NEG_INF_DIAG(n: int, device: torch.device) -> torch.Tensor:
    """Returns a diagonal matrix of size [n, n].

    The diagonal are all "-inf". This is for avoiding calculating the
    overlapped element in the Criss-Cross twice.
    """
    return torch.diag(torch.tensor(float('-inf')).to(device).repeat(n), 0)


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module.

    .. note::
        Before v1.3.13, we use a CUDA op. Since v1.3.13, we switch
        to a pure PyTorch and equivalent implementation. For more
        details, please refer to https://github.com/open-mmlab/mmcv/pull/1201.

        Speed comparison for one forward pass

        - Input size: [2,512,97,97]
        - Device: 1 NVIDIA GeForce RTX 2080 Ti

        +-----------------------+---------------+------------+---------------+
        |                       |PyTorch version|CUDA version|Relative speed |
        +=======================+===============+============+===============+
        |with torch.no_grad()   |0.00554402 s   |0.0299619 s |5.4x           |
        +-----------------------+---------------+------------+---------------+
        |no with torch.no_grad()|0.00562803 s   |0.0301349 s |5.4x           |
        +-----------------------+---------------+------------+---------------+

    Args:
        in_channels (int): Channels of the input feature map.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = Scale(0.)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward function of Criss-Cross Attention.

        Args:
            x (torch.Tensor): Input feature with the shape of
                (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output of the layer, with the shape of
            (batch_size, in_channels, height, width)
        """
        B, C, H, W = x.size()
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        energy_H = torch.einsum('bchw,bciw->bwhi', query, key) + NEG_INF_DIAG(
            H, query.device)
        energy_H = energy_H.transpose(1, 2)
        energy_W = torch.einsum('bchw,bchj->bhwj', query, key)
        attn = F.softmax(
            torch.cat([energy_H, energy_W], dim=-1), dim=-1)  # [B,H,W,(H+W)]
        out = torch.einsum('bciw,bhwi->bchw', value, attn[..., :H])
        out += torch.einsum('bchj,bhwj->bchw', value, attn[..., H:])

        out = self.gamma(out) + x
        out = out.contiguous()

        return out


# if __name__ == '__main__':
#     model = CrissCrossAttention(1024)   # 24051073
#     # model = u_segnext_b()     # 36878721
#     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print("Number of params:", n_parameters)
#     img = torch.randn(2, 1024, 256, 256)
#     out = model(img)
#     print(out.shape)
#     # for i in range(len(out)):
#     #     print(out[i].shape)