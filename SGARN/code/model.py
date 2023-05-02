import torch
from torch import nn
import torch.nn.functional as F


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[1, 2, 1],
                    [0, 0, 0],
                    [-1,-2,-1]]
        kernel_h = [[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = kernel_h
        self.weight_v = kernel_v

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v.to(x.device), padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h.to(x.device), padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v.to(x.device), padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h.to(x.device), padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v.to(x.device), padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h.to(x.device), padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Encoder(nn.Module):
    def __init__(self, in_ch=6, out_ch=48):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_r = self.relu(x1)
        x2 = self.conv2(x1_r)
        x2_r = self.relu(x2)
        x3 = self.conv3(x2_r)
        x3_r = self.relu(x3)
        out = torch.cat([x1_r,x2_r,x3_r],dim=1)

        return out

class gradient_Residual_Block(nn.Module):
    def __init__(self, i_channel=3, o_channel=3, stride=1, downsample=None):
        super(gradient_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=i_channel, out_channels=9, kernel_size=3, stride=stride, padding=1,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=o_channel, kernel_size=3, stride=1, padding=1,bias=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Grad_resnet(nn.Module):
    def __init__(self):
        super(Grad_resnet, self).__init__()
        self.resblock = gradient_Residual_Block()
        self.resblock1 = gradient_Residual_Block()
        self.conv_1X1 = nn.Conv2d(3, 3, kernel_size=1, padding=0, bias=True)

    def forward(self,x):
        x1 = self.resblock(x)
        x2 = self.resblock1(x1)
        x3 = self.conv_1X1(x2)
        return x2,x3

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, n_head=8, d_in=48, d_hidden=24):
        super(MultiHeadSelfAttentionModule, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x):
        b,c,h,w = x.size()
        x = x.view(b,c,-1)
        x = x.transpose(1,2)
        attn = self.w_2(self.relu(self.w_1(x)))
        attn = attn.transpose(1,2)
        attn = attn.view(b,self.n_head,h,w)
        attn1 = attn[:,:self.n_head//2,:,:]
        attn2 = attn[:, self.n_head//2:self.n_head, :, :]
        attn1 = self.softmax(attn1)
        attn2 = self.softmax(attn2)
        return attn1,attn2

class UniAttention(nn.Module):
    def __init__(self,channel=48, reduction=2):
        super(UniAttention, self).__init__()
        self.MTHDatt = MultiHeadSelfAttentionModule()
        self.conv_1x1_1 = nn.Conv2d(96, 48, kernel_size=1, padding=0, bias=False)
        self.conv_1x1_2 = nn.Conv2d(432, 96, kernel_size=1, padding=0, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())
        self.fc2 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self,feature1,refer):
        x = torch.cat([feature1,refer],dim=1)
        x = self.conv_1x1_1(x)
        b,c,h,w = x.size()
        x_max = self.max_pool(x).view(b, c)
        x_avg = self.avg_pool(x).view(b, c)
        y_max = self.fc1(x_max).view(b, c, 1, 1)
        att_max = x * y_max
        y_avg = self.fc2(x_avg).view(b, c, 1, 1)
        att_avg = x * y_avg
        att1 = att_avg + att_max

        MHattFeature1, MHattFeature2 = self.MTHDatt(x)
        output_list = []
        for i in range(MHattFeature1.shape[1]):
            if MHattFeature1.shape[0] == 1:
                MHattFeature1.squeeze(0)
                weighted = MHattFeature1[0][i] * feature1
            else:
                weighted = MHattFeature1[1][i] * feature1.squeeze(0)
            output_list.append(weighted)

        for i in range(MHattFeature2.shape[1]):
            if MHattFeature2.shape[0] == 1:
                MHattFeature2.squeeze(0)
                weighted = MHattFeature2[0][i] * refer
            else:
                weighted = MHattFeature2[1][i] * refer.squeeze(0)
            output_list.append(weighted)

        output = torch.cat(output_list,dim=1)
        output = torch.cat([output,att1],dim=1)

        output = self.conv_1x1_2(output)
        return output

class UniAttention_NoRefer(nn.Module):
    def __init__(self,channel=48, reduction=2):
        super(UniAttention_NoRefer, self).__init__()
        self.MTHDatt = MultiHeadSelfAttentionModule()
        self.conv_1x1_1 = nn.Conv2d(96, 48, kernel_size=1, padding=0, bias=False)
        self.conv_1x1_2 = nn.Conv2d(432, 96, kernel_size=1, padding=0, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())
        self.fc2 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self,feature1,feature2):
        x = torch.cat([feature1,feature2],dim=1)
        x = self.conv_1x1_1(x)
        b,c,h,w = x.size()
        x_max = self.max_pool(x).view(b, c)
        x_avg = self.avg_pool(x).view(b, c)
        y_max = self.fc1(x_max).view(b, c, 1, 1)
        f1_att_max = x * y_max
        y_avg = self.fc2(x_avg).view(b, c, 1, 1)
        f1_att_avg = x * y_avg
        att1 = f1_att_max + f1_att_avg
        MHattFeature1, MHattFeature2 = self.MTHDatt(x)
        output_list = []
        for i in range(MHattFeature1.shape[1]):
            if MHattFeature1.shape[0] == 1:
                MHattFeature1.squeeze(0)
                weighted = MHattFeature1[0][i] * feature1
            else:
                weighted = MHattFeature1[1][i] * feature1.squeeze(0)
            output_list.append(weighted)

        for i in range(MHattFeature2.shape[1]):
            if MHattFeature2.shape[0] == 1:
                MHattFeature2.squeeze(0)
                weighted = MHattFeature2[0][i] * feature2
            else:
                weighted = MHattFeature2[1][i] * feature2.squeeze(0)
            output_list.append(weighted)

        output = torch.cat(output_list,dim=1)
        output = torch.cat([output,att1],dim=1)

        output = self.conv_1x1_2(output)
        return output


class AttentionNetwork(nn.Module):
    def __init__(self):
        super(AttentionNetwork, self).__init__()
        self.attention_refer=UniAttention()
        self.attention_norefer=UniAttention_NoRefer()
        self.scale=Scale(1)

    def forward(self, feature1, refer, feature2):
        feature1_1 = self.attention_refer(feature1, refer)
        feature1_2 = self.attention_norefer(feature1, feature2)
        feature2_3 = self.attention_refer(feature2, refer)
        feature1_1 = self.scale(feature1_1)
        feature1_2 = self.scale(feature1_2)
        feature2_3 = self.scale(feature2_3)
        refer = self.scale(refer)
        out = torch.cat([feature1_1,feature2_3,feature1_2,refer], dim=1)
        return out

class IRDB(nn.Module):
    def __init__(self, iChannels_1X1, oChannels_1X1, growthRate,):
        super(IRDB, self).__init__()
        self.conv1 = nn.Conv2d(oChannels_1X1, growthRate, kernel_size=1, padding=0, bias=True, dilation=1)
        self.conv2 = nn.Conv2d(oChannels_1X1, growthRate, kernel_size=5, padding=2, bias=True, dilation=1)

        self.conv3 = nn.Conv2d(oChannels_1X1+growthRate*2, growthRate, kernel_size=3, padding=1, bias=True, dilation=1)
        self.conv4 = nn.Conv2d(oChannels_1X1+growthRate*2, growthRate, kernel_size=3, padding=1, bias=True, dilation=1)

        self.conv5 = nn.Conv2d(oChannels_1X1+growthRate*3, growthRate, kernel_size=5, padding=2, bias=True, dilation=1)
        self.conv6 = nn.Conv2d(oChannels_1X1+growthRate*3, growthRate, kernel_size=1, padding=0, bias=True, dilation=1)

        self.conv_1x1_1 = nn.Conv2d(iChannels_1X1, oChannels_1X1, kernel_size=1, padding=0, bias=True)
        self.conv_1x1_2 = nn.Conv2d(iChannels_1X1, oChannels_1X1, kernel_size=1, padding=0, bias=True)
        self.conv_1x1_3 = nn.Conv2d((oChannels_1X1+growthRate*3)*2, iChannels_1X1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x1 = self.conv_1x1_1(x)
        x2 = self.conv_1x1_2(x)
        x1_c1 = F.relu(self.conv1(x1))
        x2_c1 = F.relu(self.conv2(x2))
        x1_out1 = torch.cat((x1, x1_c1, x2_c1), dim=1)
        x2_out1 = torch.cat((x2, x2_c1, x1_c1), dim=1)
        x1_c2 = F.relu(self.conv3(x1_out1))
        x2_c2 = F.relu(self.conv4(x2_out1))
        x1_out2 = torch.cat((x1_c2, x2_c2, x1_c1, x1), dim=1)
        x2_out2 = torch.cat((x2_c2, x1_c2, x2_c1, x2), dim=1)
        x1_c3 = F.relu(self.conv5(x1_out2))
        x2_c3 = F.relu(self.conv6(x2_out2))
        x1_out2 = torch.cat((x1_c3, x1_c2, x1_c1, x1), dim=1)
        x2_out2 = torch.cat((x2_c3, x2_c2, x2_c1, x2), dim=1)
        out = torch.cat([x1_out2, x2_out2], dim=1)

        out = self.conv_1x1_3(out)
        out = out + x
        return out

class merger(nn.Module):
    def __init__(self):
        super(merger, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(336, 64, kernel_size=1, padding=0, bias=True)
        self.conv_1x1_2 = nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=True)
        self.conv_1x1_3 = nn.Conv2d(192, 48, kernel_size=1, padding=0, bias=True)
        self.conv_1x1_4 = nn.Conv2d(51, 32, kernel_size=1, padding=0, bias=True)
        self.conv_1x1_5 = nn.Conv2d(32, 3, kernel_size=1, padding=0, bias=True)
        self.conv_1x1_5 = nn.Conv2d(32, 3, kernel_size=1, padding=0, bias=True)
        self.conv_1x1_5 = nn.Conv2d(32, 3, kernel_size=1, padding=0, bias=True)

        self.conv1=nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(67, 48, kernel_size=3, padding=1)
        self.conv3=nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.conv4=nn.Conv2d(67, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(80, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.conv8 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(3, 3, kernel_size=3, padding=1)


        self.Get_gradient = Get_gradient()
        self.grad_resnet = Grad_resnet()
        self.IRDB1 = IRDB(iChannels_1X1=64, oChannels_1X1=32, growthRate=32)
        self.IRDB2 = IRDB(iChannels_1X1=64, oChannels_1X1=32, growthRate=32)
        self.IRDB3 = IRDB(iChannels_1X1=64, oChannels_1X1=32, growthRate=32)

    def forward(self, output, x2_enconded, x2 ):
        x = self.conv_1x1_1(output)
        x = F.relu(x)

        gradient_map1 = self.Get_gradient(x2)
        res3, res4 = self.grad_resnet(gradient_map1)

        x_DRDB1 = self.IRDB1(x)
        x_DRDB2 = self.IRDB2(x_DRDB1)
        x_DRDB3 = self.IRDB3(x_DRDB2)

        x_D = torch.cat((x_DRDB1,x_DRDB2,x_DRDB3),dim=1)
        x_D_de_ch = self.conv_1x1_3(x_D)
        x_D_de_ch = x_D_de_ch + x2_enconded
        x_D_de_ch = self.conv1(x_D_de_ch)
        x_D_de_ch = torch.cat([x_D_de_ch , res3],dim=1)
        x_D_de_ch = self.conv_1x1_4(x_D_de_ch)
        x_D_de_ch = F.relu(x_D_de_ch)
        x_D_de_ch = self.conv_1x1_5(x_D_de_ch)
        merger_output = self.conv6(x_D_de_ch)
        merger_output = torch.sigmoid(merger_output)
        return merger_output , res4


class BASENet(nn.Module):
    def __init__(self):
        super(BASENet, self).__init__()
        self.E=Encoder()
        self.M=merger()
        self.A = AttentionNetwork()


    def forward(self, x1, x2, x3):
        x1=self.E(x1)
        x2_enconded=self.E(x2)
        x3=self.E(x3)

        output = self.A(x1, x2_enconded, x3)
        merger_output , fusion_gradient=self.M(output, x2_enconded, x2)

        return merger_output, fusion_gradient
