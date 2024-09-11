import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
import yaml
from models.FasterNet import fasternet

#########################################################################################################
class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, leaky_relu=False, use_bn=True, prelu=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        if use_bn:
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
#########################################################################################################

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

        features = self.fs(rgb1,dep1,sal = mix_s2_s3, exfe = features_2, s1 = 1)

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
#########################################################################################################

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
#########################################################################################################

class SEFF(nn.Module):
    def __init__(self,inc=1029,channels=1024, r=4):
        super(SEFF, self).__init__()

        inter_channels = int(channels // r)

        self.d_conv1 = ConvBNReLU(inc, channels, ksize=1, pad=0, prelu=False)
        self.d_conv2 = ConvBNReLU(inc, channels, ksize=1, pad=0, prelu=False)

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

    def forward(self, x, dep, s=None):
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

#########################################################################################################

class DWConv(nn.Module):
    def __init__(self, channel, k=3, d=1, relu=False):
        super(DWConv, self).__init__()
        conv = [
            nn.Conv2d(channel, channel, k, 1, (k//2)*d, d, channel, False),
            nn.BatchNorm2d(channel)
        ]
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)         
                                                    
class LGB(nn.Module):
    def __init__(self,channel):
        super(LGB,self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.conv1_1 = nn.Conv2d(channel, channel*4,kernel_size=1)
        self.convdw_1_1 = nn.Conv2d(channel*4, channel,kernel_size=1)
        self.DWconv = DWConv(channel*4)
    
    def forward(self, x):
        x = self.conv1_1(x)
        x1,x2,x3,x4 = x.chunk(4, dim=1)
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)                      
        x3 = self.pool3(x3)
        x4 = self.pool4(x4)
        x_dwconv = self.DWconv(torch.cat([x1,x2,x3,x4],dim=1))
        out = self.convdw_1_1(x_dwconv)
        return out
#########################################################################################################
class MLP_4d(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)# [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C]
        x = self.proj(x)
        x= x.transpose(1,2).view(b, c, h, w)
        return x

class CFD(nn.Module):
    def __init__(self, in_channel, out_channel, out_channel_1=None):
        super(CFD, self).__init__()
        
        self.inners_a = ConvBNReLU(in_channel, out_channel // 2, ksize=1, pad=0)
        self.inners_b = ConvBNReLU(out_channel_1, out_channel // 2, ksize=1, pad=0)

        self.channel_att = nn.Sequential(nn.Linear(out_channel, out_channel), 
                                         nn.ReLU(),
                                         nn.Linear(out_channel, out_channel),
                                         nn.Sigmoid()
                                         )
        
        self.spatial_att = nn.Sequential(nn.Conv2d(out_channel, 1, kernel_size=1, stride=1, padding=0),
                                         nn.Sigmoid()
                                         )

        self.lgb = LGB(out_channel)
        self.mlp = MLP_4d(out_channel,out_channel)
        self.cbr = ConvBNReLU(out_channel, out_channel, ksize=1, pad=0)

    def forward(self, feature, exr):
        # print(feature.shape)
        # print(exr.shape)
        inner_top_down = self.inners_b(exr)
        inner_lateral = self.inners_a(feature)
        high_up = F.interpolate(inner_top_down, size=inner_lateral.shape[2:], mode='bilinear', align_corners=False)
        
        mulres = high_up * inner_lateral

        fuse1 = torch.cat((high_up, mulres), dim=1)
        fuse2 = torch.cat((mulres, inner_lateral), dim=1)
        fuse = fuse1 + fuse2

        lgb = self.lgb(fuse)
        ca =  self.channel_att(fuse.mean(dim=2).mean(dim=2)).unsqueeze(dim=2).unsqueeze(dim=2) 
        sa =  self.spatial_att(fuse)
        
        final =  self.mlp(lgb * ca)
        final =  self.cbr(sa * final)
        # print(final.shape)
        return final

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

        self.deco_1 = CFD(in_channels[-1],out_channels[-1], out_channel_1=out_channels[-1])
        self.deco_2 = CFD(in_channels[-2],out_channels[-2], out_channel_1=out_channels[-1])
        self.deco_3 = CFD(in_channels[-3],out_channels[-3], out_channel_1=out_channels[-2])
        self.deco_4 = CFD(in_channels[-4],out_channels[-4], out_channel_1=out_channels[-3])
    
    def forward(self,features,depth,flag=1,sal=None, exfe=None, s1=0):       
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
