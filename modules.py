import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_dct as dct

class CompConv(nn.Module):
    def __init__(self, ni, no, kernel_size, stride=1, padding=0, bias=True, g=4, r=2, progressive=False):
        super(CompConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ni = ni
        self.no = no
		self.r = r * ((no * ni * kernel_size**2)**(0.5) / 64) + 1 if progressive else r
        self.index = nn.Parameter(torch.IntTensor((no * ni * kernel_size**2) // g), requires_grad=False)
        self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(g, int(((no * self.ni * kernel_size**2) // g)/self.r)), mode='fan_out', nonlinearity='relu'))
        self.bias = nn.Parameter(nn.init.zeros_(torch.Tensor(no))) if bias else None
    def forward(self, x):
        filt = dct.idct(F.pad(self.weight, (0, self.index.size(0) - self.weight.size(1))), norm='ortho')
        filt = filt[:, self.index.long()]
        filt = torch.reshape(filt, (self.no, self.ni, self.kernel_size, self.kernel_size))

        x = F.conv2d(x, filt, bias=self.bias, stride=self.stride, padding=self.padding)
        return x


class CompLinear(nn.Module):
    def __init__(self, ni, no, bias=True, g=4, r=2, progressive=False):
        super(CompLinear, self).__init__()
        self.ni = ni
        self.no = no
		self.r = r * ((no * ni)**(0.5) / 64) + 1 if progressive else r
        self.index = nn.Parameter(torch.IntTensor((no * ni) // g), requires_grad=False)
        self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(g, int(((no * self.ni) // g)/self.r)), mode='fan_out', nonlinearity='relu'))
        self.bias = nn.Parameter(nn.init.zeros_(torch.Tensor(no))) if bias else None
    def forward(self, x):
        filt = dct.idct(F.pad(self.weight, (0, self.index.size(0) - self.weight.size(1))), norm='ortho')
        filt = filt[:, self.index.long()]
        filt = torch.reshape(filt, (self.no, self.ni))

        x = F.linear(x, filt, bias=self.bias)
        return x