from torch import nn
import torch
from torchsparse import nn as spnn
import numpy as np
import open3d as o3d
from torchsparse.utils.quantize import sparse_quantize
import torchsparse.nn as spnn
from torchsparse.tensor import SparseTensor


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


###################################  feature net  ######################################
class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
            ConvBnReLU(3, 8, 3, 1, 1),
            ConvBnReLU(8, 8, 3, 1, 1))

        self.conv1 = nn.Sequential(
            ConvBnReLU(8, 16, 5, 2, 2),
            ConvBnReLU(16, 16, 3, 1, 1),
            ConvBnReLU(16, 16, 3, 1, 1))

        self.conv2 = nn.Sequential(
            ConvBnReLU(16, 32, 5, 2, 2),
            ConvBnReLU(32, 32, 3, 1, 1),
            ConvBnReLU(32, 32, 3, 1, 1))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return torch.nn.functional.interpolate(x, scale_factor=2,
                                               mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        conv0 = self.conv0(x)  # (B, 8, H, W)
        conv1 = self.conv1(conv0)  # (B, 16, H//2, W//2)
        conv2 = self.conv2(conv1)  # (B, 32, H//4, W//4)
        feat2 = self.toplayer(conv2)  # (B, 32, H//4, W//4)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))  # (B, 32, H//2, W//2)
        feat0 = self._upsample_add(feat1, self.lat0(conv0))  # (B, 32, H, W)

        # reduce output channels
        feat1 = self.smooth1(feat1)  # (B, 16, H//2, W//2)
        feat0 = self.smooth0(feat0)  # (B, 8, H, W)

        # feats = {"level_0": feat0,
        #          "level_1": feat1,
        #          "level_2": feat2}

        return [feat2, feat1, feat0]  # coarser to finer features


class BasicSparseConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicSparseDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)

class SparseResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1), spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out
    
class SparseCostRegNet(nn.Module):

    def __init__(self, d_in,d_out=8):
        super(SparseCostRegNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.conv0 = BasicSparseConvolutionBlock(d_in, d_out)

        self.conv1 = BasicSparseConvolutionBlock(d_out, 16, stride=2)
        self.conv2 = BasicSparseConvolutionBlock(16, 16)

        self.conv3 = BasicSparseConvolutionBlock(16, 32, stride=2)
        self.conv4 = BasicSparseConvolutionBlock(32, 32)

        self.conv5 = BasicSparseConvolutionBlock(32, 64, stride=2)
        self.conv6 = BasicSparseConvolutionBlock(64, 64)

        self.conv7 = BasicSparseDeconvolutionBlock(64, 32, ks=3, stride=2)

        self.conv9 = BasicSparseDeconvolutionBlock(32, 16, ks=3, stride=2)

        self.conv11 = BasicSparseDeconvolutionBlock(16, d_out, ks=3, stride=2)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        return x.F
    
def sparse_to_dense_volume(sparse_tensor, coords,vol_dim, default_val=0):
    c = sparse_tensor.shape[-1]
    coords = coords.to(torch.int64)
    ## clamp the coords to prevent the data overflow
    coords[:, 0] = coords[:, 0].clamp(0,vol_dim[0]-1)
    coords[:, 1] = coords[:, 1].clamp(0,vol_dim[1]-1)
    coords[:, 2] = coords[:, 2].clamp(0,vol_dim[2]-1)
    device = sparse_tensor.device
    dense = torch.full([vol_dim[0], vol_dim[1], vol_dim[2], c], float(default_val), device=device) #type: ignore
    dense[coords[:, 0], coords[:, 1], coords[:, 2]] = sparse_tensor
    return dense

def construct_sparse_tensor(raw_coords, feats, Bbx_min: torch.Tensor, Bbx_max: torch.Tensor, voxel_size=0.1):
    X_MIN, X_MAX = Bbx_min[0], Bbx_max[0]
    Y_MIN, Y_MAX = Bbx_min[1], Bbx_max[1]
    Z_MIN, Z_MAX = Bbx_min[2], Bbx_max[2]

    if isinstance(raw_coords, torch.Tensor) or isinstance(feats, torch.Tensor):
        raw_coords = raw_coords.cpu().numpy()
        feats = feats.cpu().numpy()

    bbx_max = np.array([X_MAX,Y_MAX,Z_MAX])
    bbx_min = np.array([X_MIN,Y_MIN,Z_MIN])
    vol_dim = (bbx_max - bbx_min) / 0.1
    vol_dim = vol_dim.astype(int).tolist()

    raw_coords -= np.array([X_MIN,Y_MIN,Z_MIN]).astype(int)
    coords, indices = sparse_quantize(raw_coords, voxel_size, return_index=True)  ## voxelize the pnt to discrete formation
    coords = torch.tensor(coords, dtype=torch.int).cuda()
    zeros = torch.zeros(coords.shape[0], 1).cuda()
    ## Note: [B,X,Y,Z] in Torch sparsev 2.1
    coords = torch.cat((zeros,coords), dim=1).to(torch.int32)  

    feats = torch.tensor(feats[indices], dtype=torch.float).cuda()
    sparse_feat = SparseTensor(feats,coords=coords)
    return sparse_feat, vol_dim, coords[:,1:]