""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#====================================================================
#=========================U-Net======================================
#====================================================================
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    def __init__(self, channel, k_size=3):
        super(GrayscaleAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.linear = nn.Linear(256, 256)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        is_int = 1
        if c != 1:
            raise NotImplementedError()
        if x.max() <= 1:
            is_int = 0
            x.data *= 255
            x = x.long()

        x_onehot = torch.zeros([b, c, 256, h, w])
        try:
            x_onehot = x_onehot.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # print(x)
        # print(x_onehot)
        x_onehot = x_onehot.scatter_(2, x.unsqueeze(2).long(), 1) #[b,c,d,w,h]
        a = x_onehot.squeeze(1) #[b,d,w,h]
        y = self.avg_pool(a) #[b,d,1,1]
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.linear(y.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
        y = self.sigmoid(y)
        a = a * y.expand_as(a)
        x_onehot = a.unsqueeze(1)
        x = torch.argmax(x_onehot, dim=2).float()
        return x if is_int==1 else x.data/255

#====================================================================
#==========================Transformer===============================
#====================================================================
class MultiHeadDense(nn.Module):
    def __init__(self, d):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
    
    def forward(self, x):
        # x:[b, h*w, d]
        # x = torch.bmm(x, self.weight)
        x = F.linear(x, self.weight)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
    
    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D

        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()

class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel)
        self.key = MultiHeadDense(channel)
        self.value = MultiHeadDense(channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.positional_encoding_2d(c, h, w)
        x = x + pe
        x = x.reshape(b, c, h*w).permute(0, 2, 1) #[b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(c))#[b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        return x

class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, channelY, channelS):
        super(MultiHeadCrossAttention, self).__init__()
        self.Sconv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True)
        )
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True)
        )
        self.query = MultiHeadDense(channelS)
        self.key = MultiHeadDense(channelS)
        self.value = MultiHeadDense(channelS)
        self.conv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size()
        Yb, Yc, Yh, Yw = Y.size()
        Spe = self.positional_encoding_2d(Sc, Sh, Sw)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh*Yw).permute(0, 2, 1)
        V = self.value(S1)
        Ype = self.positional_encoding_2d(Yc, Yh, Yw)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh*Yw).permute(0, 2, 1)
        Y2 = self.Yconv2(Y)
        Q = self.query(Y1)
        K = self.key(Y1)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Sc))
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)
        Z = self.conv(x)
        Z = Z * S
        Z = torch.cat([Z, Y2], dim=1)
        return Z



class TransformerUp(nn.Module):
    def __init__(self, Ychannels, Schannels):
        super(TransformerUp, self).__init__()
        self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels)
        self.conv = nn.Sequential(
            nn.Conv2d(Ychannels, Schannels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(Schannels, Schannels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, Y, S):
        x = self.MHCA(Y, S)
        x = self.conv(x)
        return x

