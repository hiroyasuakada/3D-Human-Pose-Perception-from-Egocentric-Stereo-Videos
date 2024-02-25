import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils import weight_norm
import functools
from torchvision import models
import torch.nn.functional as F
from torch.optim import lr_scheduler
from collections import OrderedDict
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .network_transformer import PoseTransformerQADF


######################################################################################
# Functions
######################################################################################

def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.2, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def init_weights(net, opt, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.uniform_(m.weight.data, gain, 1.0)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def print_network_param(net, name):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('total number of parameters of {}: {:.3f} M'.format(name, num_params / 1e6))


def init_net(net, opt, init_type='normal', gpu_ids=[], init_ImageNet=True):

    if init_ImageNet is False:
        init_weights(net, opt, init_type)
    else:
        init_weights(net.after_backbone, opt, init_type)
        print('   ... also using ImageNet initialization for the backbone')

    if opt.use_ddp:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = net.to(opt.local_rank)
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[opt.local_rank],
            find_unused_parameters=True
            )
    elif len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()

    return net


def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def unfreeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()
        m.weight.requires_grad = True
        m.bias.requires_grad = True

def freeze_bn_affine(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.requires_grad = False
        m.bias.requires_grad = False


######################################################################################
# Define networks
######################################################################################


def define_HeatMap(opt, model):

    if model == "unrealego2_pose_qa_df":
        net = HeatMap_UnrealEgo_Shared(opt)

    else:
        raise ValueError('Model [%s] not recognized.' % opt.model)

    print_network_param(net, 'HeatMap Estimator: {}'.format(model))

    return init_net(net, opt, opt.init_type, opt.gpu_ids, opt.init_ImageNet)


def define_Pose(opt, model):

    if model == "unrealego2_pose_qa_df":
        net = TemporalPoseEstimatorQADF(opt)

    else:
        raise ValueError('Model [%s] not recognized.' % opt.model)

    print_network_param(net, 'PoseEstimator for {}'.format(model))

    return init_net(net, opt, opt.init_type, opt.gpu_ids, False)


######################################################################################
# Basic Operation
######################################################################################


def make_conv_layer(in_channels, out_channels, kernel_size, stride, padding, with_bn=True):
    conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding)
    bn = torch.nn.BatchNorm2d(num_features=out_channels)
    relu = torch.nn.LeakyReLU(negative_slope=0.2)
    if with_bn:
        return torch.nn.Sequential(conv, bn, relu)
    else:
        return torch.nn.Sequential(conv, relu)

def make_deconv_layer(in_channels, out_channels, kernel_size, stride, padding, with_bn=True):
    conv = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding)
    bn = torch.nn.BatchNorm2d(num_features=out_channels)
    relu = torch.nn.LeakyReLU(negative_slope=0.2)
    if with_bn:
        return torch.nn.Sequential(conv, bn, relu)
    else:
        return torch.nn.Sequential(conv, relu)

def make_fc_layer(in_feature, out_feature, with_relu=True, with_bn=True):
    modules = OrderedDict()
    fc = torch.nn.Linear(in_feature, out_feature)
    modules['fc'] = fc
    bn = torch.nn.BatchNorm1d(num_features=out_feature)
    relu = torch.nn.LeakyReLU(negative_slope=0.2)

    if with_bn is True:
        modules['bn'] = bn
    else:
        print('no bn')

    if with_relu is True:
        modules['relu'] = relu
    else:
        print('no pose relu')

    return torch.nn.Sequential(modules)

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


######################################################################################
# Network structure
######################################################################################


############# Heatmap Estimator from UnrealEgo paper #############

class HeatMap_UnrealEgo_Shared(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(HeatMap_UnrealEgo_Shared, self).__init__()

        self.backbone = HeatMap_UnrealEgo_Shared_Backbone(opt, model_name=model_name)
        self.after_backbone = HeatMap_UnrealEgo_AfterBackbone(opt, model_name=model_name)

    def forward(self, input_left, input_right, return_feat=False):
        x_left, x_right = self.backbone(input_left, input_right)
        output = self.after_backbone(x_left, x_right, return_feat=return_feat)

        return output


class HeatMap_UnrealEgo_Shared_Backbone(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(HeatMap_UnrealEgo_Shared_Backbone, self).__init__()

        self.backbone = Encoder_Block(opt, model_name=model_name)

    def forward(self, input_left, input_right):
        output_left = self.backbone(input_left)
        output_right = self.backbone(input_right)

        return output_left, output_right

class Encoder_Block(nn.Module):
    def __init__(self, opt, model_name='resnet18'):
        super(Encoder_Block, self).__init__()

        if opt.init_ImageNet:
            if model_name == 'resnet18':
                self.backbone = models.resnet18('ResNet18_Weights.DEFAULT')
            elif model_name == "resnet34":
                self.backbone = models.resnet34('ResNet34_Weights.DEFAULT')
            elif model_name == "resnet50":
                self.backbone = models.resnet50('ResNet50_Weights.DEFAULT')
            elif model_name == "resnet101":
                self.backbone = models.resnet101('ResNet101_Weights.DEFAULT')
            else:
                raise NotImplementedError('model type [%s] is invalid', model_name)
        else:
            if model_name == 'resnet18':
                self.backbone = models.resnet18()
            elif model_name == "resnet34":
                self.backbone = models.resnet34()
            elif model_name == "resnet50":
                self.backbone = models.resnet50()
            elif model_name == "resnet101":
                self.backbone = models.resnet101()
            else:
                raise NotImplementedError('model type [%s] is invalid', model_name)


        self.base_layers = list(self.backbone.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

    def forward(self, input):

        feat0 = self.layer0(input)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        output = [input, feat0, feat1, feat2, feat3, feat4]

        return output


class HeatMap_UnrealEgo_AfterBackbone(nn.Module):
    def __init__(self, opt, model_name="resnet18"):
        super(HeatMap_UnrealEgo_AfterBackbone, self).__init__()

        if model_name == 'resnet18':
            feature_scale = 1
        elif model_name == "resnet34":
            feature_scale = 1
        elif model_name == "resnet50":
            feature_scale = 4
        elif model_name == "resnet101":
            feature_scale = 4
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)


        self.num_heatmap = opt.num_heatmap

        self.layer1_1x1 = convrelu(128 * feature_scale, 128 * feature_scale, 1, 0)
        self.layer2_1x1 = convrelu(256 * feature_scale, 256 * feature_scale, 1, 0)
        self.layer3_1x1 = convrelu(512 * feature_scale, 516 * feature_scale, 1, 0)
        self.layer4_1x1 = convrelu(1024 * feature_scale, 1024 * feature_scale, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(516 * feature_scale + 1024 * feature_scale, 1024 * feature_scale, 3, 1)
        self.conv_up2 = convrelu(256 * feature_scale + 1024 * feature_scale, 512 * feature_scale, 3, 1)
        self.conv_up1 = convrelu(128 * feature_scale + 512 * feature_scale, 512 * feature_scale, 3, 1)

        self.conv_heatmap = nn.Conv2d(512 * feature_scale, self.num_heatmap * 2, 1)

    def forward(self, list_input_left, list_input_right, return_feat=False):
        list_feature_stereo = [
            torch.cat([list_input_left[id], list_input_right[id]], dim=1) for id in range(len(list_input_left))
        ]

        input = list_feature_stereo[0]
        feat0 = list_feature_stereo[1]
        feat1 = list_feature_stereo[2]
        feat2 = list_feature_stereo[3]
        feat3 = list_feature_stereo[4]
        feat4 = list_feature_stereo[5]

        layer4 = self.layer4_1x1(feat4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(feat3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(feat2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(feat1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        output = self.conv_heatmap(x)

        if return_feat:
            return [output, feat4]

        return output


####################### Pose Estimator #######################


class TemporalPoseEstimatorQADF(nn.Module):

    def __init__(self, opt):
        super(TemporalPoseEstimatorQADF, self).__init__()

        self.seq_len = opt.seq_len
        self.num_heatmap = opt.num_heatmap
        self.pred_seq_pose = opt.pred_seq_pose
        self.use_single_query = opt.use_single_query
        self.use_depth_padding_mask = opt.use_depth_padding_mask
        self.num_depth_channels = 2

        self.with_bn = True
        self.with_pose_relu = True

        self.conv1 = make_conv_layer(in_channels=self.num_heatmap, out_channels=64, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv2 = make_conv_layer(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv3 = make_conv_layer(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv4 = make_conv_layer(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)

        self.conv1_depth = make_conv_layer(in_channels=2, out_channels=16, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv2_depth = make_conv_layer(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv3_depth = make_conv_layer(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv4_depth = make_conv_layer(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)
        self.conv5_depth = make_conv_layer(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, with_bn=self.with_bn)

        self.transformer = PoseTransformerQADF(
            query_adaptation=opt.query_adaptation,
            seq_len=opt.seq_len,
            d_input=512,
            d_input_depth=256,
            d_model=opt.hidden_dim,
            dropout=opt.dropout,
            nhead=opt.nheads,
            dim_feedforward=opt.dim_feedforward,
            num_encoder_layers=opt.enc_layers,
            num_decoder_layers=opt.dec_layers,
            normalize_before=opt.pre_norm,
            return_intermediate_dec=True,
            use_tgt_mask=opt.use_tgt_mask,
            use_single_query=self.use_single_query,
            use_depth_padding_mask=self.use_depth_padding_mask,
        )

        self.pose_fc1 = make_fc_layer(in_feature=256, out_feature=128, with_relu=self.with_pose_relu, with_bn=False)
        self.pose_fc2 = make_fc_layer(in_feature=128, out_feature=64, with_relu=self.with_pose_relu, with_bn=False)
        self.pose_fc3 = torch.nn.Linear(64, 3)


    def forward(self,
                input_video_left,
                input_video_right,
                video_feature_stereo,
                sfm_depth_left,
                sfm_depth_mask_left,
                sfm_depth_right,
                sfm_depth_mask_right,
                torch_labels,
        ):


        # RGB Stereo: left (B*L, C, H, W) + right (B*L, C, H, W) -> stereo (2*B*L, C, H, W)
        input_video_stereo = torch.cat([input_video_left, input_video_right], dim=0)

        # encode heatmap and depth
        x = self.conv1(input_video_stereo)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Depth Left and Right: depth (B*L, C, H, W) + mask (B*L, C, H, W) -> stereo (B*L, 2*C, H, W)
        depth_video_left = torch.cat([sfm_depth_left, sfm_depth_mask_left], dim=1)
        depth_video_right = torch.cat([sfm_depth_right, sfm_depth_mask_right], dim=1)

        # Depth Stereo: depth (B*L, 2*C, H, W) + mask (B*L, 2*C, H, W) -> stereo (2*B*L, 2C, H, W)
        depth_video_stereo = torch.cat([depth_video_left, depth_video_right], dim=0)

        d = self.conv1_depth(depth_video_stereo)
        d = self.conv2_depth(d)
        d = self.conv3_depth(d)
        d = self.conv4_depth(d)
        d = self.conv5_depth(d)

        # Process in Transformer Decoder
        x = self.transformer(x, video_feature_stereo, d, torch_labels)[0]  # pose feature (B, L * Pose 16, C)

        if self.use_single_query:
            x_pose = self.pose_fc1(x)
            x_pose = self.pose_fc2(x_pose)
            x_pose = self.pose_fc3(x_pose)
            x_pose = rearrange(x_pose, "b (l p) c -> b l p c", l=int(1), p=int(self.num_heatmap + 1))

        else:
            if self.pred_seq_pose:
                x_pose = self.pose_fc1(x)
                x_pose = self.pose_fc2(x_pose)
                x_pose = self.pose_fc3(x_pose)
                x_pose = rearrange(x_pose, "b (l p) c -> b l p c", l=self.seq_len, p=int(self.num_heatmap + 1))
            else:
                x = rearrange(x, "b (l p) c -> b l p c", l=self.seq_len, p=int(self.num_heatmap + 1))
                x = x[:, -1]
                x_pose = self.pose_fc1(x)
                x_pose = self.pose_fc2(x_pose)
                x_pose = self.pose_fc3(x_pose)
                x_pose = rearrange(x_pose, "b (l p) c -> b l p c", l=int(1), p=int(self.num_heatmap + 1))

        return x_pose  # pose (B, L, Pose 16, 3) or pose (B, 1, Pose 16, 3)


