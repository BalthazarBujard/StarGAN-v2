"""
Most of this code is inspired/taken from: 

https://github.com/protossw512/AdaptiveWingLoss 

and 

https://github.com/clovaai/stargan-v2/blob/master/core/wing.py for the specific heatmap preprocessing elements unavailable from the article
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import namedtuple
from munch import Munch
from copy import deepcopy
from .coord_conv import CoordConvTh


def conv3x3(in_planes, out_planes, strd=1, padding=1,
            bias=False,dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias,
                     dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4),
                             padding=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4),
                             padding=1, dilation=1)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(x_dim=64, y_dim=64,
                                     with_r=True, with_boundary=True,
                                     in_channels=256, first_one=first_one,
                                     out_channels=256,
                                     kernel_size=1,
                                     stride=1, padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x, heatmap):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel

#Face Allignment Network
class FAN(nn.Module):
    def __init__(self, num_modules=1, end_relu=False, num_landmarks=98, pretrained_file=None): 
        #needs the same 98 landmarks as stargan git for compatibility with preprocessing, default is 68
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.end_relu = end_relu
        self.num_landmarks = num_landmarks

        self.conv1 = CoordConvTh(x_dim=256, y_dim=256,
                                     with_r=True, with_boundary=False,
                                     in_channels=3, out_channels=64,
                                     kernel_size=7,
                                     stride=2, padding=3)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            if hg_module == 0:
                first_one = True
            else:
                first_one = False
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256,
                                                            first_one))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            num_landmarks+1, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(num_landmarks+1,
                                                                 256, kernel_size=1, stride=1, padding=0))

        #load pretrained model
        if pretrained_file != None:
            checkpoint = torch.load(pretrained_file)
            model_weights = self.state_dict()
            #only update model weights form state dict in checkpoint file
            model_weights.update({k: v for k, v in checkpoint['state_dict'].items()
                              if k in model_weights})
            self.load_state_dict(model_weights)

            #self.load_state_dict(torch.load(pretrained_file))

    def forward(self, x):
        x, _ = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        # x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        boundary_channels = []
        tmp_out = None
        for i in range(self.num_modules):
            hg, boundary_channel = self._modules['m' + str(i)](previous,
                                                               tmp_out)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            if self.end_relu:
                tmp_out = F.relu(tmp_out) # HACK: Added relu
            outputs.append(tmp_out)
            boundary_channels.append(boundary_channel)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs, boundary_channels

    @torch.no_grad()
    def get_heatmap(self, x, b_preprocess=True):
        ''' outputs 0-1 normalized heatmap '''
        x = F.interpolate(x, size=256, mode='bilinear') #
        x_01 = x*0.5 + 0.5 #(x+1)/2 passer de -1,1 a 0,1
        outputs, _ = self(x_01) #get landmarks
        heatmaps = outputs[-1][:, :-1, :, :] #take last element of outputs (batch dim) and from that tensor take everything except last channel (:-1)
        scale_factor = x.size(2) // heatmaps.size(2)
        if b_preprocess:
            heatmaps = F.interpolate(heatmaps, scale_factor=scale_factor,
                                     mode='bilinear', align_corners=True)
            heatmaps = preprocess(heatmaps) #if the size of the heatmap is different as input imgs -> process 
        return heatmaps

""" 
Taken from starganv2 git
"""
def normalize(x, eps=1e-6):
    """Apply min-max normalization."""
    x = x.contiguous()
    N, C, H, W = x.size()
    x_ = x.view(N*C, -1) #check for every pixel in inputs
    max_val = torch.max(x_, dim=1, keepdim=True)[0] #max
    min_val = torch.min(x_, dim=1, keepdim=True)[0] #min
    x_ = (x_ - min_val) / (max_val - min_val + eps) #normalize pixels
    out = x_.view(N, C, H, W) #output as same shape 
    return out


def truncate(x, thres=0.1):
    """Remove small values in heatmaps."""
    return torch.where(x < thres, torch.zeros_like(x), x)


def resize(x, p=2):
    """Resize heatmaps."""
    return x**p


def shift(x, N):
    """Shift N pixels up or down."""
    up = N >= 0
    N = abs(N)
    _, _, H, W = x.size()
    head = torch.arange(N)
    tail = torch.arange(H-N)

    if up:
        head = torch.arange(H-N)+N
        tail = torch.arange(N)
    else:
        head = torch.arange(N) + (H-N)
        tail = torch.arange(H-N)

    # permutation indices
    perm = torch.cat([head, tail]).to(x.device)
    out = x[:, :, perm, :]
    return out

#landmarks indexes
IDXPAIR = namedtuple('IDXPAIR', 'start end')
index_map = Munch(chin=IDXPAIR(0 + 8, 33 - 8),
                  eyebrows=IDXPAIR(33, 51),
                  eyebrowsedges=IDXPAIR(33, 46),
                  nose=IDXPAIR(51, 55),
                  nostrils=IDXPAIR(55, 60),
                  eyes=IDXPAIR(60, 76),
                  lipedges=IDXPAIR(76, 82),
                  lipupper=IDXPAIR(77, 82),
                  liplower=IDXPAIR(83, 88),
                  lipinner=IDXPAIR(88, 96))
OPPAIR = namedtuple('OPPAIR', 'shift resize')

# short analysis : translate and rescale heatmap landmarks and does a second copy without facelines and mouth for deeper G layers
def preprocess(x):
    """Preprocess 98-dimensional heatmaps."""
    N, C, H, W = x.size()
    x = truncate(x)
    x = normalize(x)

    sw = H // 256
    operations = Munch(chin=OPPAIR(0, 3),
                       eyebrows=OPPAIR(-7*sw, 2),
                       nostrils=OPPAIR(8*sw, 4),
                       lipupper=OPPAIR(-8*sw, 4),
                       liplower=OPPAIR(8*sw, 4),
                       lipinner=OPPAIR(-2*sw, 3))

    for part, ops in operations.items():
        start, end = index_map[part]
        x[:, start:end] = resize(shift(x[:, start:end], ops.shift), ops.resize)

    zero_out = torch.cat([torch.arange(0, index_map.chin.start),
                          torch.arange(index_map.chin.end, 33),
                          torch.LongTensor([index_map.eyebrowsedges.start,
                                            index_map.eyebrowsedges.end,
                                            index_map.lipedges.start,
                                            index_map.lipedges.end])])
    x[:, zero_out] = 0

    start, end = index_map.nose
    x[:, start+1:end] = shift(x[:, start+1:end], 4*sw)
    x[:, start:end] = resize(x[:, start:end], 1)

    start, end = index_map.eyes
    x[:, start:end] = resize(x[:, start:end], 1)
    x[:, start:end] = resize(shift(x[:, start:end], -8), 3) + \
        shift(x[:, start:end], -24)

    # Second-level mask
    x2 = deepcopy(x)
    x2[:, index_map.chin.start:index_map.chin.end] = 0  # start:end was 0:33
    x2[:, index_map.lipedges.start:index_map.lipinner.end] = 0  # start:end was 76:96
    x2[:, index_map.eyebrows.start:index_map.eyebrows.end] = 0  # start:end was 33:51

    x = torch.sum(x, dim=1, keepdim=True)  # (N, 1, H, W)
    x2 = torch.sum(x2, dim=1, keepdim=True)  # mask without faceline and mouth

    x[x != x] = 0  # set nan to zero
    x2[x != x] = 0  # set nan to zero
    return x.clamp_(0, 1), x2.clamp_(0, 1)
