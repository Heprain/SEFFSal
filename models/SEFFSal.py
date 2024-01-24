# import sys
# sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
import math
import yaml

from models.FasterNet import fasternet

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s

class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, leaky_relu=False, use_bn=True, frozen=False, spectral_norm=False, prelu=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            else:
                self.bn = BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            if leaky_relu is True:
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif prelu is True:
                self.act = nn.PReLU(nOut)
            else:
                self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ResidualConvBlock, self).__init__()
        self.conv = ConvBNReLU(nIn, nOut, ksize=ksize, stride=stride, pad=pad,
                               dilation=dilation, groups=groups, bias=bias,
                               use_relu=use_relu, use_bn=use_bn, frozen=frozen)
        self.residual_conv = ConvBNReLU(nIn, nOut, ksize=1, stride=stride, pad=0,
                               dilation=1, groups=groups, bias=bias,
                               use_relu=False, use_bn=use_bn, frozen=frozen)

    def forward(self, x):
        x = self.conv(x) + self.residual_conv(x)
        return x

class ReceptiveConv(nn.Module):
    def __init__(self, inplanes, planes, baseWidth=24, scale=4, dilation=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(ReceptiveConv, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, self.width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width*scale)
        #self.nums = 1 if scale == 1 else scale - 1
        self.nums = scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dilation = [1] * self.nums if dilation is None else dilation
        for i in range(self.nums):
            self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, \
                    padding=dilation[i], dilation=dilation[i], bias=False))
            self.bns.append(nn.BatchNorm2d(self.width))

        self.conv3 = nn.Conv2d(self.width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        #if self.scale > 1:
        #    out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)

        return out

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, residual=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, ksize=1, pad=0))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SEFFSal(nn.Module):
    def __init__(self, net_size):
        super(SEFFSal, self).__init__()

        self.s1_feats = generate_feats(size=net_size)
        self.s2_feats = generate_feats(size=net_size)
        self.s3_feats = generate_feats(size=net_size)
        
        if net_size == 'm':
            enc_channels=[144, 288, 576, 1152]
            dec_channels=[144, 288, 576, 1152]
        elif net_size == 's':
            enc_channels=[128, 256, 512, 1024]
            dec_channels=[128, 256, 512, 1024]
        elif net_size == 't':
            enc_channels=[96, 192, 384, 768]
            dec_channels=[96, 192, 384, 768]
        else:
            raise Exception("The information you provided is incorrect!")

        self.fs = decoder(enc_channels, dec_channels)
        self.fs2 = decoder(enc_channels, dec_channels)
        self.fs3 = decoder(enc_channels, dec_channels,need=False)

        self.cls1 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)
        self.cls2 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)
        self.cls4 = nn.Conv2d(dec_channels[3], 1, 1, stride=1, padding=0)

        self.cls1_2 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)
        self.cls2_2 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3_2 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)
        self.cls4_2 = nn.Conv2d(dec_channels[3], 1, 1, stride=1, padding=0)

        self.cls1_3 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)
        self.cls2_3 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3_3 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)
        self.cls4_3 = nn.Conv2d(dec_channels[3], 1, 1, stride=1, padding=0)
        self.change = nn.Conv2d(8, 4, 1, stride=1, padding=0)
        
        self.c1 = nn.Conv2d(4, 1, 1, stride=1, padding=0)
        self.c2 = nn.Conv2d(4, 1, 1, stride=1, padding=0)
        self.c3 = nn.Conv2d(4, 1, 1, stride=1, padding=0)

    def forward(self, input, depth=None):
        s1_r = input
        s1_d = depth
      
        down = nn.Upsample(scale_factor=0.5, mode = 'bilinear',align_corners=True)
       
        s2_r = down(s1_r)  
        s2_d = down(s1_d)

        s3_r = down(s2_r)   
        s3_d = down(s2_d)

        ##-------------------------------------------
        ##-------------- Scale 3---------------------
        ##-------------------------------------------
        # generate backbone features
        rgb3, dep3 = self.s3_feats(s3_r,s3_d)
        
        #decoder
        features_3 = self.fs3(rgb3,dep3,flag = 0)         

        saliency_maps_3 = []
        for idx, feature in enumerate(features_3[:4]):
            saliency_maps_3.append(F.interpolate(
                    getattr(self, 'cls' + str(idx + 1) + '_3')(feature),
                    input.shape[2:],
                    mode='bilinear',
                    align_corners=False)
            )  

        saliency_maps_3 = torch.sigmoid(torch.cat(saliency_maps_3, dim=1)) 
        

        ##-------------------------------------------
        ##-------------- Scale 2---------------------
        ##-------------------------------------------
        # generate backbone features
        rgb2, dep2= self.s2_feats(s2_r, s2_d)

        #decoder
        features_2 = self.fs2(rgb2,dep2,sal = saliency_maps_3, exfe = features_3)
    
        saliency_maps_2 = []
        for idx, feature in enumerate(features_2[:4]):
            saliency_maps_2.append(F.interpolate(
                    getattr(self, 'cls' + str(idx + 1) + '_2')(feature),
                    input.shape[2:],
                    mode='bilinear',
                    align_corners=False)
            )
            
        saliency_maps_2 = torch.sigmoid(torch.cat(saliency_maps_2, dim=1))

        ##-------------------------------------------
        ##-------------- Scale 1---------------------
        ##-------------------------------------------
        # generate backbone features
        rgb1, dep1= self.s1_feats(s1_r, s1_d)

        mix_s2_s3 = torch.cat((saliency_maps_2,saliency_maps_3),1)
        mix_s2_s3 = self.change(mix_s2_s3)

        features = self.fs(rgb1,dep1,sal = mix_s2_s3, exfe = features_2)

        saliency_maps = []
        for idx, feature in enumerate(features[:4]):
            saliency_maps.append(F.interpolate(
                    getattr(self, 'cls' + str(idx + 1))(feature),
                    input.shape[2:],
                    mode='bilinear',
                    align_corners=False)
            )

        saliency_maps = torch.sigmoid(torch.cat(saliency_maps, dim=1))

        return saliency_maps, saliency_maps_2, saliency_maps_3

class generate_feats(nn.Module):
    def __init__(self, size, pretrained=True):
        super(generate_feats, self).__init__()
        src = './cfg/'
        with open(src + 'fasternet_' + size +'.yaml', 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        self.rgbnet = fasternet(size, pretrained, **yaml_data)
        self.depthnet = fasternet(size, pretrained=False, in_chans=1,**yaml_data)
    
    def forward(self,img,depth):
        #layer Pe
        x = self.rgbnet.patch_embed(img)
        y = self.depthnet.patch_embed(depth)
        outs_r = []
        outs_d = []

        #layer 0
        x = self.rgbnet.stages[0](x)
        y = self.depthnet.stages[0](y)
        outs_r.append(x)
        outs_d.append(y)

        #layer 1
        x = self.rgbnet.stages[1](x)
        y = self.depthnet.stages[1](y)

        #layer 2
        x = self.rgbnet.stages[2](x)
        y = self.depthnet.stages[2](y)
        outs_r.append(x)
        outs_d.append(y)

        #layer 3
        x = self.rgbnet.stages[3](x)
        y = self.depthnet.stages[3](y)

        #layer 4
        x = self.rgbnet.stages[4](x)
        y = self.depthnet.stages[4](y)
        outs_r.append(x)
        outs_d.append(y)


        #layer 5
        x = self.rgbnet.stages[5](x)
        y = self.depthnet.stages[5](y)

        #layer 6
        x = self.rgbnet.stages[6](x)
        y = self.depthnet.stages[6](y)
        outs_r.append(x)
        outs_d.append(y)

        return outs_r, outs_d

class SEFF(nn.Module):
    def __init__(self,inc=1029,channels=1024, r=4):
        super(SEFF, self).__init__()

        inter_channels = int(channels // r)
        self.d_conv1 = InvertedResidual(inc, channels, residual=True)  #CBR-CBR-CB
        self.d_conv2 = InvertedResidual(inc, channels, residual=True)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, dep, s=None,flag=0):
        if flag:  # flag=1 two inputs
            dep = F.interpolate(dep, x.shape[2:], mode = 'bilinear',align_corners=True)
            dep = self.d_conv1(dep)
            xa = x + dep
        else:   # flag=0   three inputs
            if s is not None:  # input=F1 F2 S
                s1 = F.interpolate(s, dep.shape[2:], mode = 'bilinear',align_corners=True)
                s2 = F.interpolate(s, x.shape[2:], mode = 'bilinear',align_corners=True)
                dep = torch.cat([dep, s1],1)
                x = torch.cat([x, s2],1)
                dep = self.d_conv1(dep)
                x = self.d_conv2(x)
                dep = F.interpolate(dep, x.shape[2:], mode = 'bilinear',align_corners=True)
                xa = x + dep
            else: # input= F1 F2
                xa = x + dep
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        fu = self.sigmoid(xlg)
        xo = 2 * x * fu + 2 * dep * (1 - fu)
        return xo

class CPR(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, dilation=[1,2,3], residual=True):
        super(CPR, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        self.conv1 = ConvBNReLU(inp, hidden_dim, ksize=1, pad=0, prelu=False)

        self.hidden_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[0], groups=hidden_dim, dilation=dilation[0])
        self.hidden_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[1], groups=hidden_dim, dilation=dilation[1])
        self.hidden_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[2], groups=hidden_dim, dilation=dilation[2])
        self.hidden_bnact = nn.Sequential(nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        m = self.conv1(x)
        m = self.hidden_conv1(m) + self.hidden_conv2(m) + self.hidden_conv3(m)
        m = self.hidden_bnact(m)
        if self.use_res_connect:
            return x + self.out_conv(m)
        else:
            return self.out_conv(m)

class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, input_num=2):
        super(Fusion, self).__init__()
        if input_num == 2:
            self.channel_att = nn.Sequential(nn.Linear(in_channels, in_channels),
                                             nn.ReLU(),
                                             nn.Linear(in_channels, in_channels),
                                             nn.Sigmoid()
                                             )
        self.fuse = nn.Sequential( CPR(in_channels, in_channels, expand_ratio=expansion, residual=True),
                                      ConvBNReLU(in_channels, in_channels, ksize=1, pad=0, stride=1)
                                      )

    def forward(self, low, high=None):
        if high is None:
            final = self.fuse(low)
        else:
            high_up = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
            fuse = torch.cat((high_up, low), dim=1)
            final = self.channel_att(fuse.mean(dim=2).mean(dim=2)).unsqueeze(dim=2).unsqueeze(dim=2) * self.fuse(fuse)

        return final

class CPRDecoder(nn.Module):
    def __init__(self, in_channel, out_channel, out_channel_1=None, begin=False):
        super(CPRDecoder, self).__init__()
        if begin:
            self.inners_a = ConvBNReLU(in_channel, out_channel, ksize=1, pad=0)
            self.fuse = Fusion(out_channel, out_channel, input_num=1)
        else:
            self.inners_a = ConvBNReLU(in_channel, out_channel // 2, ksize=1, pad=0)
            self.inners_b = ConvBNReLU(out_channel_1, out_channel // 2, ksize=1, pad=0)
            self.fuse = Fusion(out_channel, out_channel)

    def forward(self, feature, exr = None):
        if exr is not None:
            inner_top_down = self.inners_b(exr)
            inner_lateral = self.inners_a(feature)
            stage_result = self.fuse(inner_lateral, inner_top_down) 
        else:
            stage_result = self.fuse(self.inners_a(feature))
        
        return stage_result

    
class decoder(nn.Module):   #self.fs2 = decoder(enc_channels, dec_channels)
    def __init__(self, in_channels, out_channels, need = True):
        super(decoder, self).__init__()
        if need:  #s1,s2
            self.seff_1 = SEFF(inc= in_channels[0] + 4, channels=in_channels[0]) #r、s
            self.seff_2 = SEFF(inc= in_channels[1] + 4, channels=in_channels[1])
            self.seff_3 = SEFF(inc= in_channels[2] + 4, channels=in_channels[2])
            self.seff_4 = SEFF(inc= in_channels[3] + 4, channels=in_channels[3])  # r、d、s
            self.seff_5 = SEFF(inc= in_channels[3] + 4, channels=in_channels[3])  # r、d、s
        
        else:  #s3
            self.seff =  SEFF(inc= in_channels[-1], channels=in_channels[-1])

        self.deco_1 = CPRDecoder(in_channels[-1],out_channels[-1], out_channel_1=out_channels[-1])
        self.deco_2 = CPRDecoder(in_channels[-2],out_channels[-2], out_channel_1=out_channels[-1])
        self.deco_3 = CPRDecoder(in_channels[-3],out_channels[-3], out_channel_1=out_channels[-2])
        self.deco_4 = CPRDecoder(in_channels[-4],out_channels[-4], out_channel_1=out_channels[-3])
    
    def forward(self,features,depth,flag=1,sal=None, exfe=None):       
        conv1, conv2, conv3, conv4 = features
        results = []
        if not flag: #s3   features_3 = self.fs3(rgb3,dep3,flag = 0)
            fu = self.seff(conv4,depth[-1])
            stage_result = self.deco_1(conv4,fu) #Fcpr31
            results = [stage_result]

            stage_result = self.deco_2(conv3,stage_result) #Fcpr32
            results.insert(0, stage_result)

            stage_result = self.deco_3(conv2,stage_result) #Fcpr33
            results.insert(0, stage_result)

            stage_result = self.deco_4(conv1,stage_result) #Fcpr34
            results.insert(0, stage_result)
        else: #s1,2  features_2 = self.fs2(rgb2,dep2,sal = saliency_maps_3, exfe = features_3) 
            fu = self.seff_5(conv4, depth[-1], sal)

            result1 = self.deco_1(conv4,fu)
            stage_result = self.seff_4(result1,exfe[-1],sal)
            results = [stage_result]

            result2 = self.deco_2(conv3,stage_result + result1)
            stage_result = self.seff_3(result2,exfe[-2],sal)
            results.insert(0, stage_result)

            result3 = self.deco_3(conv2,stage_result + result2)
            stage_result = self.seff_2(result3,exfe[-3],sal)
            results.insert(0, stage_result)

            result4 = self.deco_4(conv1,stage_result + result3)
            stage_result = self.seff_1(result4,exfe[-4],sal)
            results.insert(0, stage_result)

        return results
