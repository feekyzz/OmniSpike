# from visualizer import get_local
import torch
import torchinfo
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
import warnings
# from visualizer import get_local
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from typing import Tuple, Optional


from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors
import math
# __all__ = ('MS_GetT','MS_CancelT', 'MS_ConvBlock','MS_Block','MS_DownSampling',
#            'MS_StandardConv','SpikeSPPF','SpikeConv','MS_Concat','SpikeDetect'
#            ,'Ann_ConvBlock','Ann_DownSampling','Ann_StandardConv','Ann_SPPF','MS_C2f',
#            'Conv_1','BasicBlock_1','BasicBlock_2','Concat_res2','Sample','MS_FullConvBlock','MS_ConvBlock_resnet50','MS_AllConvBlock','MS_ConvBlock_res2net')



'''
此版本为0614之前的版本，发现这个版本有很多不是脉冲输入，原始代码将mem_update()注释掉了。

'''
decay = 0.25  # 0.25 # decay constants




class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)

        self.act = act
        self.qtrick = MultiSpike4()  # change the max value

    def forward(self, x):

        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]

            else:
                mem = x[i]
            spike = self.qtrick(mem)

            mem_old = mem.clone()
            output[i] = spike
        # print(output[0][0][0][0])
        return output

class MultiSpike8(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类

    class quant8(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=8))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 8] = 0
            return grad_input

    def forward(self, x):
#         print(self.quant8.apply(x))
        return self.quant8.apply(x)

class MultiSpike4(nn.Module):

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)

class MultiSpike2(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类

    class quant2(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=2))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 2] = 0
            return grad_input

    def forward(self, x):
        return self.quant2.apply(x)

class MultiSpike1(nn.Module):

    class quant1(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=1))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 1] = 0
            return grad_input

    def forward(self, x):
        return self.quant1.apply(x)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



@torch.jit.script
def jit_mul(x, y):
    return x.mul(y)

@torch.jit.script
def jit_sum(x):
    return x.sum(dim=[-1, -2], keepdim=True)

class SpikeDFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)  #[0,1,2,...,15]
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1)) #这里不是脉冲驱动的，但是是整数乘法
        self.c1 = c1  #本质上就是个加权和。输入是每个格子的概率(小数)，权重是每个格子的位置(整数)
        self.lif = mem_update()


    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)  # 原版

class SpikeDetect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(SpikeConv(x, c2, 3), SpikeConv(c2, c2, 3), SpikeConvWithoutBN(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(SpikeConv(x, c3, 3), SpikeConv(c3, c3, 3), SpikeConvWithoutBN(c3, self.nc, 1)) for x in ch)
        self.dfl = SpikeDFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].mean(0).shape  # BCHW  推理：[1，2，64，32，84]  这里必须mean0，否则推理时用到shape会导致报错
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 2)
            x[i] = x[i].mean(0)  #[2，144，32，684]  #这个地方有时候全是1.之后debug看看
        bs = x[0].shape[0]
        preds = {
            "boxes": torch.cat([xi[:, : self.reg_max * 4].view(bs, 4 * self.reg_max, -1) for xi in x], 2),
            "scores": torch.cat([xi[:, self.reg_max * 4 :].view(bs, self.nc, -1) for xi in x], 2),
            "feats": x,
        }
        if self.training:
            return preds
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) #box: [B,reg_max * 4,anchors]
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, preds)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].conv.bias.data[:] = 1.0  # box
            b[-1].conv.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


            
class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean
                        * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class RepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size = 3,
            bias=False,
            group = 1
    ):
        super().__init__()
        padding = int((kernel_size-1)/2)
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            # mem_update(), #11111
            nn.Conv2d(in_channel, in_channel, kernel_size, 1,0, groups=in_channel, bias=False),  #这里也是分组卷积
            # mem_update(),  #11111
            nn.Conv2d(in_channel, out_channel, 1,  1,0, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

    
class SepRepConv(nn.Module): 
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size = 3,
            bias=False,
            group = 1
    ):
        super().__init__()
        padding = int((kernel_size-1)/2)
        # hidden_channel = in_channel
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1,0, groups=group, bias=False),  #这里也是分组卷积
            # mem_update(), #11111
            nn.Conv2d(out_channel, out_channel, kernel_size,  1,0, groups=out_channel, bias=False),
        )


        self.body = nn.Sequential(bn, conv3x3)

    def forward(self, x):
        return self.body(x)

class SepAllConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act2_layer=nn.Identity,
                 bias=False,
                 kernel_size=3,  #7,3
                 padding=1):
        super().__init__()
        padding = int((kernel_size -1)/2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size, #7*7
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
#         self.pwconv3 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias,groups=1)
        self.pwconv3=SepRepConv(med_channels, dim)  #这里将sepconv最后一个卷积替换为重参数化卷积  大概提0.5个点，可以保留

        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)


        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
#         print("x.shape:",x.shape)
        x = self.lif1(x) #x1_lif:0.2328  x2_lif:0.0493  这里x2的均值偏小，因此其经过bn和lif后也偏小，发放率比较低；而x1均值偏大，因此发放率也高
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)  # flatten：从第0维开始，展开到第一维
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act2_layer=nn.Identity,
                 bias=False,
                 kernel_size=3,  #7,3
                 padding=1):
        super().__init__()
        padding = int((kernel_size -1)/2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size, #7*7
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
#         self.pwconv3 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias,groups=1)
        self.pwconv3=SepRepConv(med_channels, dim)  #这里将sepconv最后一个卷积替换为重参数化卷积  大概提0.5个点，可以保留

        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)


        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
#         print("x.shape:",x.shape)
        x = self.lif1(x) #x1_lif:0.2328  x2_lif:0.0493  这里x2的均值偏小，因此其经过bn和lif后也偏小，发放率比较低；而x1均值偏大，因此发放率也高
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)  # flatten：从第0维开始，展开到第一维
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x






class MS_ConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7 ,full=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.full =full
        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)  #内部扩张2倍
        self.mlp_ratio = mlp_ratio
        
        self.lif1 = mem_update()
        self.lif2 = mem_update()

        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio)) #137以外的模型，在第一个block不做分组

        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))  # 这里可以进行改进

        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)  # 这里可以进行改进




    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  #sepconv  pw+dw+pw

        x_feat = x

        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
            #repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x

    
class MS_AllConvBlock(nn.Module):  # standard conv
    def __init__(self, input_dim, mlp_ratio=4.,sep_kernel_size = 7 ,group=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim,kernel_size= sep_kernel_size)

        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio),3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim,3)



    # @get_local('x_feat')
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x  #sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = x_feat + x

        return x

class MS_StandardConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):  # in_channels(out_channels), 内部扩张比例
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.lif = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.bn(self.conv(self.lif(x).flatten(0, 1))).reshape(T, B, self.c2, int(H / self.s), int(W / self.s))
        return x

class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()

        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = mem_update()
        # self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        T, B, _, _, _ = x.shape


        if hasattr(self, "encode_lif"): #如果不是第一层
            # x_pool = self.pool(x)
            x = self.encode_lif(x)


        x = self.encode_conv(x.flatten(0, 1))
        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x

class MS_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=4):
        super().__init__()
        self.T = T
        self.in_channels = in_channels

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        return x

class MS_CancelT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=2):
        super().__init__()
        self.T = T

    def forward(self, x):
        x = x.mean(0)
        return x

class SpikeConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = mem_update()
        self.bn = nn.BatchNorm2d(c2)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        return x

class SpikeConvWithoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        self.lif = mem_update()

        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x

class SpikeSPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = SpikeConv(c1, c_, 1, 1)
        self.cv2 = SpikeConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            return self.cv2(torch.cat((x, y1, y2, y3), 2))



class MS_Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):  # 这里输入x是一个list
        for i in range(len(x)):
            if x[i].dim() == 5:
                x[i] = x[i].mean(0)
        return torch.cat(x, self.d)




def _grid2seq(x: Tensor, region_size: Tuple[int, int], num_heads: int):
    """
    x: [B, C, H, W]
    return:
        seq: [B, num_heads, n_region, region_token_num, head_dim]
    """
    B, C, H, W = x.size()
    region_h, region_w = H // region_size[0], W // region_size[1]
    x = x.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1])
    x = torch.einsum('bmdhpwq->bmhwpqd', x).flatten(2, 3).flatten(-3, -2)
    return x, region_h, region_w


def _seq2grid(x: Tensor, region_h: int, region_w: int, region_size: Tuple[int, int]):
    """
    x: [B, num_heads, n_region, region_token_num, head_dim]
    return:
        grid: [B, C, H, W]
    """
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    x = x.view(bs, nhead, region_h, region_w, region_size[0], region_size[1], head_dim)
    x = torch.einsum('bmhwpqd->bmdhpwq', x).reshape(
        bs, nhead * head_dim, region_h * region_size[0], region_w * region_size[1]
    )
    return x


def regional_routing_attention_torch(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    region_graph: LongTensor,
    region_size: Tuple[int, int],
    kv_region_size: Optional[Tuple[int, int]] = None,
    auto_pad: bool = True,
):
    """
    query, key, value: [B, C, H, W]
    region_graph: [B, num_heads, q_nregion, topk]
    return:
        output: [B, C, H, W]
        attn:   [B, num_heads, q_nregion, q_region_token_num, topk * kv_region_token_num]
    """
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()

    q_pad_b = q_pad_r = kv_pad_b = kv_pad_r = 0

    if auto_pad:
        _, _, Hq, Wq = query.size()
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        if q_pad_b > 0 or q_pad_r > 0:
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b))

        _, _, Hk, Wk = key.size()
        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        if kv_pad_b > 0 or kv_pad_r > 0:
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b))
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b))
    else:
        Hq, Wq = query.shape[-2:]
        Hk, Wk = key.shape[-2:]

    query, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    bs, nhead, kv_nregion, kv_region_token_num, head_dim = key.size()

    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1).expand(
        -1, -1, -1, -1, kv_region_token_num, head_dim
    )

    key_g = torch.gather(
        key.view(bs, nhead, 1, kv_nregion, kv_region_token_num, head_dim).expand(
            -1, -1, q_nregion, -1, -1, -1
        ),
        dim=3,
        index=broadcasted_region_graph
    )  # [B, nhead, q_nregion, topk, kv_region_token_num, head_dim]

    value_g = torch.gather(
        value.view(bs, nhead, 1, kv_nregion, kv_region_token_num, head_dim).expand(
            -1, -1, q_nregion, -1, -1, -1
        ),
        dim=3,
        index=broadcasted_region_graph
    )

    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    output = attn @ value_g.flatten(-3, -2)

    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        output = output[:, :, :Hq, :Wq]

    return output, attn


class BiLevelRoutingAttention_SNN(nn.Module):
    """
    基于 MS_StandardConv 的 SNN 适配版 Bi-Level Routing Attention

    输入:
        x: [T, B, C, H, W]
    输出:
        out: [T, B, C, H, W]
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        n_win=7,
        topk=4,
        qk_scale=None,
        side_dwconv=3,
        auto_pad=True,
        route_mode='time_avg',   # 'time_avg' / 'last'
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_win = n_win
        self.topk = topk
        self.auto_pad = auto_pad
        self.route_mode = route_mode

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.head_dim = dim // num_heads
        self.scale = qk_scale or (self.head_dim ** -0.5)

        # ===== 用你的 MS_StandardConv 做 Q/K/V =====
        self.q_proj = MS_StandardConv(dim, dim, k=1, s=1)
        self.k_proj = MS_StandardConv(dim, dim, k=1, s=1)
        self.v_proj = MS_StandardConv(dim, dim, k=1, s=1)

        # 局部位置编码 LePE
        self.lepe = SpikeConv(dim, dim, k=side_dwconv, s=1, g=dim) if side_dwconv > 0 else nn.Identity()

        # 输出投影，也用你的 MS_StandardConv
        self.out_proj = MS_StandardConv(dim, dim, k=1, s=1)

    def _build_region_graph(self, q: Tensor, k: Tensor):
        """
        q, k: [B, C, H, W]
        return:
            idx_r: [B, num_heads, n_region, topk]
            region_size: (rh, rw)
        """
        B, C, H, W = q.shape

        # 每边分成 n_win 个窗口
        region_h = max(H // self.n_win, 1)
        region_w = max(W // self.n_win, 1)
        region_size = (region_h, region_w)

        q_r = F.avg_pool2d(q, kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k, kernel_size=region_size, ceil_mode=True, count_include_pad=False)

        q_r = q_r.permute(0, 2, 3, 1).flatten(1, 2)   # [B, n_region, C]
        k_r = k_r.flatten(2, 3)                       # [B, C, n_region]

        a_r = q_r @ k_r                               # [B, n_region, n_region]
        topk = min(self.topk, a_r.shape[-1])
        _, idx_r = torch.topk(a_r, k=topk, dim=-1)   # [B, n_region, topk]
        idx_r = idx_r.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        return idx_r, region_size

    def forward(self, x, ret_attn_mask=False):
        """
        x: [T, B, C, H, W]
        """
        T, B, C, H, W = x.shape

        # 1) Spike Q/K/V
        q = self.q_proj(x)   # [T,B,C,H,W]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) 构建区域路由图：用时间聚合后的特征更稳定
        if self.route_mode == 'time_avg':
            q_route = q.mean(0)   # [B,C,H,W]
            k_route = k.mean(0)
        elif self.route_mode == 'last':
            q_route = q[-1]
            k_route = k[-1]
        else:
            raise ValueError(f'Unsupported route_mode: {self.route_mode}')

        idx_r, region_size = self._build_region_graph(q_route, k_route)

        # 3) 每个时间步分别做稀疏 attention
        lepe = self.lepe(v)
        out_list = []
        attn_list = []

        for t in range(T):
            qt = q[t]   # [B,C,H,W]
            kt = k[t]
            vt = v[t]

            out_t, attn_t = regional_routing_attention_torch(
                query=qt,
                key=kt,
                value=vt,
                scale=self.scale,
                region_graph=idx_r,
                region_size=region_size,
                auto_pad=self.auto_pad,
            )

            out_t = out_t + lepe[t]
            out_list.append(out_t)

            if ret_attn_mask:
                attn_list.append(attn_t)

        out = torch.stack(out_list, dim=0)  # [T,B,C,H,W]

        # 4) 再经过 spike 输出投影
        out = self.out_proj(out)

        if ret_attn_mask:
            return out, attn_list
        return out
