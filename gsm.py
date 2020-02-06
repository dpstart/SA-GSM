# PyTorch GSM implementation by Swathikiran Sudhakaran

import torch
from torch import nn
from torch.cuda import FloatTensor as ftens
import sys

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

class GSM(nn.Module):
    def __init__(self, fPlane, num_segments=3):
        super(GSM, self).__init__()

        self.conv3D = nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1,
                                padding=(1, 1, 1), groups=2)
        nn.init.constant_(self.conv3D.weight, 0)
        nn.init.constant_(self.conv3D.bias, 0)
        self.tanh = nn.Tanh()
        self.fPlane = fPlane
        self.num_segments = num_segments
        self.bn = nn.BatchNorm3d(num_features=fPlane)
        self.relu = nn.ReLU()
        self.projector = ProjectorBlock(128, 512)
        self.attn = LinearAttentionBlock(512)

    def lshift_zeroPad(self, x):
        return torch.cat((x[:,:,1:], ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0)), dim=2)
    def rshift_zeroPad(self, x):
        return torch.cat((ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0), x[:,:,:-1]), dim=2)

    def forward(self, x):
        batchSize = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        assert  shape[0] == self.fPlane
        #print("Initial shape", x.shape)
        x = x.view(batchSize, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()
        #print("Shape after view", x.shape)
        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)

        # Spatial gating
        gate = self.tanh(self.conv3D(x_bn_relu))
       # print("Gate shape", gate.shape)
        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)
        #print("gate group shape", gate_group1.shape)
        x_group1 = x[:, :self.fPlane // 2]
        x_group2 = x[:, self.fPlane // 2:]
       # print("x group shape", x_group1.shape)
        y_group1 = gate_group1 * x_group1
        y_group2 = gate_group2 * x_group2

        r_group1 = x_group1 - y_group1
        r_group2 = x_group2 - y_group2

        y_group1 = self.lshift_zeroPad(y_group1) + r_group1
        y_group2 = self.rshift_zeroPad(y_group2) + r_group2

        #print("after shift and pad", y_group1.shape)

        y_group1 = y_group1.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4,
                                                                                                        5)
        y_group2 = y_group2.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4,
                                                                                                        5)
        #print("after view", y_group1.shape)

        y = torch.cat((y_group1.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:]),
                       y_group2.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:])), dim=1)
        out = y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize*self.num_segments, *shape)
        #print("out", out.shape)

        _, out_attn = self.attn(self.projector(out), out)

        return out_attn
